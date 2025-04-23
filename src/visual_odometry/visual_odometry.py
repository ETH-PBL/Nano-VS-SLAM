import torch
import kornia
import cv2
import numpy as np
from visual_odometry.feature_matcher import BfFeatureMatcher
import time
import kornia.feature as KF
from lightglue.lightglue import LightGlue
from lightglue.lightglue_configs import get_light_glue_config
from visual_odometry.frontend import KP2DtinyFrontend
from omegaconf import OmegaConf
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {execution_time} seconds")
        return result
    return wrapper

def get_matches_scores(kpts0, kpts1, matches0, mscores0):
    m0 = matches0 > -1
    m1 = matches0[m0]
    pts0 = kpts0[m0]
    pts1 = kpts1[m1]
    scores = mscores0[m0]
    return pts0, pts1, scores

def get_matching_keypoints(kp1, kp2, idxs):
    mkpts1 = kp1[idxs[:, 0]]
    mkpts2 = kp2[idxs[:, 1]]
    return mkpts1, mkpts2

def convert_superpts_to_keypoints(pts, size=1):
    kps = []
    if pts is not None: 
        # convert matrix [Nx2] of pts into list of keypoints 
        kps = [ cv2.KeyPoint(p[0], p[1], size=size, response=p[2]) for p in pts ]                            
    return kps 


class VisualOdometryStats:
    def __init__(self):
        self.stats = {
            'num_matches': [],
            'num_outliers': [],
            'num_inliers': [],
            'network_inference_timing': [],  # Add timing stats for network inference
            'pose_estimation_timing': [],  # Add timing stats for pose estimation
        }


    def update(self, keyword, value):
        if keyword in self.stats.keys():
            self.stats[keyword].append(value)
        else:
            raise ValueError("Invalid keyword")

    def get_stats(self):
        mean_stats = {}
        for keyword, values in self.stats.items():
            mean_stats[keyword] = np.mean(values)
        return mean_stats
    
    def get_mean(self, keyword):
        return np.mean(np.asarray(self.stats[keyword]))

class VisualOdometry(object):
    default_conf = {

        "plot": True,

        "semantic_matching": False,
        "device": "cuda",
        "extractor":{
            "method": "kp2dtiny",
            "classes_to_filter": [ 0],
            "config": "A",
            "weights_path": "./checkpoints/tiny_A_CS.ckpt",
            #"weights_path": "./checkpoints/baseline_v4_pytorch.ckpt",
            "semantic_filter": False,
            "debug": True,
            "nn_thresh": 0.7,
            "v3": False,
            "nClasses": 19,
        },

        "use_light_glue": False,
        "light_glue_path": "./checkpoints/light_glue_tiny_S.ckpt",

        "lg_threshold": 0.5, # only for keypointnet + lightglue
        "top_k_points": 6000,
        "top_k_matches": 2000,
    }
    def __init__(self, cam, image_shape, new_size, conf={}):
        """
        Initializes the VisualOdometry class. This class is used to evaluate the performance of our tiny feature detection models.

        Args:
            cam: The camera object used for visual odometry.
            image_shape: The shape of the input images.
            new_size: The desired size for resizing the images (default: (256, 1024)).
            method: The method to use for feature extraction and matching (default: "kp2dtiny").
            plot: Whether to plot the feature matches (default: False).
            semantic_filter: Whether to apply semantic filtering to the feature matches (default: True).
            semantic_matching: Whether to use semantic matching (default: False).
        """
        self.conf = OmegaConf.merge(self.default_conf, conf)
        self.prev_image = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.device = self.conf['device']
        self.semantic_matching = self.conf['semantic_matching']
        self.use_lg = self.conf['use_light_glue']

        self.new_size = new_size
        if self.new_size is not None:
            self.scale_x = self.new_size[1] / float(image_shape[1])
            self.scale_y = self.new_size[0] / float(image_shape[0])

        self.cur_R = np.eye(3, 3)  # current rotation
        self.cur_t = np.zeros((3, 1))  # current translation
        self.R_est = []
        self.t_est = []
        self.cam = cam
        self.method = self.conf['extractor']['method']
        self.stats = VisualOdometryStats()
        self.lg_threshold = self.conf['lg_threshold']
        # Initialize the LOFTR model
        if self.method == "loftr":
            self.loftr = kornia.feature.LoFTR('outdoor').to(self.device)
            self.gray = True
        elif self.method == "kp2dtiny" or self.method == "keypointnet":
            self.net = KP2DtinyFrontend(
                self.new_size,
                **self.conf['extractor'],
            )
            self.matcher = BfFeatureMatcher(norm_type=cv2.NORM_L2)
            self.gray = False
        if self.use_lg:
            if self.method == "keypointnet":
                self.lg = KF.LightGlue("superpoint").eval().to(self.device)
            else:
                self.lg = LightGlue(get_light_glue_config(self.conf['kp2dtiny_config']), self.conf['light_glue_path']).eval().to(self.device)

    def get_info(self):
        info = {'method': self.method, 'new_size': self.new_size, 'device': self.device}
        if self.method == "kp2dtiny":
            info['kp2dtiny'] = self.net.get_info()
        elif self.method == "keypointnet":
            info['keypointnet'] = self.net.get_info()
        return info
    
    def init(self, image):
        """
        Initializes the visual odometry module with an input image.

        Args:
            image (PIL.Image.Image): The input image.

        Returns:
            None
        """
        image_tensor = kornia.image_to_tensor(image).float() / 255.0
        if self.new_size is not None:
            image_tensor = kornia.geometry.transform.resize(image_tensor, size=self.new_size).to(self.device)
        self.prev_image = image_tensor

        if self.method == "kp2dtiny" or self.method=='keypointnet':
            kps_cur, feat_cur, seg_cur = self.net.run(image_tensor)
            self.prev_keypoints = kps_cur
            self.prev_descriptors = feat_cur
            self.prev_seg = seg_cur

    @timing_decorator
    def inference(self,image_tensor):
        kps_cur, feat_cur, seg_cur = self.net.run(image_tensor)
        return kps_cur, feat_cur, seg_cur
    @timing_decorator
    def match(self, image_tensor, kps_cur, feat_cur, seg_cur):

        if self.semantic_matching:
            kps0, kps1 = self.match_semantic(kps_cur, feat_cur, seg_cur)


        elif self.use_lg:
            # idxs0,idxs1 = self.matcher.match( self.prev_descriptors, feat_cur)
            kpts0_temp = torch.tensor(self.prev_keypoints).unsqueeze(0).float().to(self.device)

            kpts1_temp = torch.tensor(kps_cur).unsqueeze(0).float().to(self.device)

            image_size = torch.tensor(image_tensor.shape[1:]).view(1, 2)
            if self.method == "keypointnet":
                kpts0_temp = kpts0_temp / torch.tensor([self.new_size[1], self.new_size[0]]).unsqueeze(
                    0).float().to(self.device)
                kpts1_temp = kpts1_temp / torch.tensor([self.new_size[1], self.new_size[0]]).unsqueeze(
                    0).float().to(self.device)
                image0 = {"keypoints": kpts0_temp,
                          "descriptors": torch.tensor(self.prev_descriptors).unsqueeze(0).to(self.device),
                          'image_size': image_size}
                image1 = {"keypoints": kpts1_temp, "descriptors": torch.tensor(feat_cur).unsqueeze(0).to(self.device),
                          'image_size': image_size}
                with torch.no_grad():
                    out = self.lg({'image0': image0, 'image1': image1})
                idxs = out['matches'][0].detach().cpu().numpy()
                scores = out['scores'][0].detach().cpu().numpy()
                idxs = idxs[scores > self.lg_threshold]
                kps0 = np.asarray(self.prev_keypoints[idxs[:, 0], :])
                kps1 = np.asarray(kps_cur[idxs[:, 1], :])
            elif self.method == "kp2dtiny":
                kpts0_temp = kpts0_temp / torch.tensor([self.new_size[1], self.new_size[0]]).unsqueeze(
                    0).float().to(self.device)
                kpts1_temp = kpts1_temp / torch.tensor([self.new_size[1], self.new_size[0]]).unsqueeze(
                    0).float().to(self.device)
                data = {"keypoints0": kpts0_temp,
                        "descriptors0": torch.tensor(self.prev_descriptors).unsqueeze(0).to(self.device),
                        'view0': {'image_size': image_size}, "keypoints1": kpts1_temp,
                        "descriptors1": torch.tensor(feat_cur).unsqueeze(0).to(self.device),
                        'view1': {'image_size': image_size}}
                with torch.no_grad():
                    out = self.lg(data)
                m0, scores0 = out["matches0"].detach().cpu(), out["matching_scores0"].detach().cpu()
                kps0, kps1, scores = get_matches_scores(self.prev_keypoints, kps_cur, m0[0], scores0[0])
                if scores.__len__() > self.conf['top_k_matches'] and self.conf['top_k_matches'] > 0:
                    _, idxs = scores.topk(self.conf['top_k_matches'])
                    kps0 = kps0[idxs]
                    kps1 = kps1[idxs]
                kps0, kps1 = np.asarray(kps0), np.asarray(kps1)
        else:
            idxs0, idxs1, score = self.matcher.match(self.prev_descriptors, feat_cur)
            score = np.asarray(score)
            kps0 = np.asarray(self.prev_keypoints[idxs0, :])
            kps1 = np.asarray(kps_cur[idxs1, :])
            if score.__len__() > self.conf['top_k_matches'] and self.conf['top_k_matches'] > 0:
                top_k_idxs = np.argpartition(score, self.conf['top_k_matches'])[:self.conf['top_k_matches']]

                kps0 = kps0[top_k_idxs]
                kps1 = kps1[top_k_idxs]
        return kps0, kps1
    @timing_decorator
    def process_image(self, image, scale=1.0, conf= 0.9):
        # Convert the image to tensor
        image_tensor = kornia.image_to_tensor(image).float() / 255.0
        if self.new_size is not None:
            image_tensor = kornia.geometry.transform.resize(image_tensor, size= self.new_size).to(self.device)
        
        # calculaet resizing factors
        start_time = time.time()
        #image_tensor = image_tensor
        if self.method == "loftr":
            
            input = {"image0": self.prev_image.unsqueeze(0), "image1": image_tensor.unsqueeze(0)}
            # Extract features from the current image using LOFTR
            with torch.no_grad():
                out = self.loftr(input)

            kps0 = out["keypoints0"]
            kps1 = out["keypoints1"]
            
            kps0 = kps0[out['confidence'] > conf].cpu().detach().numpy().squeeze()
            kps1 = kps1[out['confidence'] > conf].cpu().detach().numpy().squeeze()
            # apply resizing factors to the keypoints
            
            
        elif self.method == "kp2dtiny" or self.method == "keypointnet":
            kps_cur, feat_cur, seg_cur = self.inference(image_tensor)
            kps0, kps1 = self.match(image_tensor, kps_cur, feat_cur, seg_cur)

        inference_image_time = time.time() - start_time
        self.stats.update('network_inference_timing', inference_image_time)
        self.stats.update('num_matches',kps0.shape[0])

            
        if self.new_size is not None:
            kps0[:,0] = kps0[:,0]/self.scale_x
            kps0[:,1] = kps0[:,1]/self.scale_y
            
            kps1[:,0] = kps1[:,0]/self.scale_x
            kps1[:,1] = kps1[:,1]/self.scale_y

        self.kps0 = kps0
        self.kps1 = kps1

        start_time = time.time()
        R, t = self.estimatePose(kps0, kps1)
        #R, t = np.eye(3), np.zeros((3, 1))
        self.stats.update('pose_estimation_timing', time.time() - start_time)
        self.R_est.append(R)
        self.t_est.append(t)
        self.cur_t = self.cur_t + scale*self.cur_R.dot(t)
        self.cur_R = self.cur_R.dot(R)

        # Update the previous image, keypoints, and descriptors<
        self.prev_image = image_tensor
        if self.method == "kp2dtiny" or self.method == "keypointnet":
            self.prev_keypoints = kps_cur
            self.prev_descriptors = feat_cur
            self.prev_seg = seg_cur
        return self.cur_t
    
    def match_semantic(self, kps_cur, feat_cur, seg_cur):
        """
        Matches keypoints and descriptors between the previous frame and the current frame based on semantic segmentation.

        Args:
            kps_cur (numpy.ndarray): Keypoints of the current frame.
            feat_cur (numpy.ndarray): Descriptors of the current frame.
            seg_cur (numpy.ndarray): Semantic segmentation of the current frame.

        Returns:
            numpy.ndarray: Matched keypoints from the previous frame.
            numpy.ndarray: Matched keypoints from the current frame.
        """
        kps0 = []
        kps1 = []
        for class_id in range(28):
            class_idxs0 = np.where(self.prev_seg == class_id)[0]
            class_idxs1 = np.where(seg_cur == class_id)[0]
            class_kps0 = self.prev_keypoints[class_idxs0]
            class_kps1 = kps_cur[class_idxs1]
            try:
                idxs0, idxs1 = self.matcher.match(self.prev_descriptors[class_idxs0], feat_cur[class_idxs1])
                class_kps0 = np.asarray(class_kps0[idxs0])
                class_kps1 = np.asarray(class_kps1[idxs1])

                kps0.extend(class_kps0)
                kps1.extend(class_kps1)
            except:
                print("No matches found for class ", class_id)
        kps0 = np.asarray(kps0)
        kps1 = np.asarray(kps1)
        return kps0, kps1
    @timing_decorator
    def estimatePose(self, kps_ref, kps_cur):
        kp_ref_u = self.cam.undistort_points(kps_ref)	
        kp_cur_u = self.cam.undistort_points(kps_cur)	        
        self.kpn_ref = self.cam.unproject_points(kp_ref_u)
        self.kpn_cur = self.cam.unproject_points(kp_cur_u)
        ransac_method = None 
        try: 
            ransac_method = cv2.USAC_MSAC 
        except: 
            ransac_method = cv2.RANSAC
        # the essential matrix algorithm is more robust since it uses the five-point algorithm solver by D. Nister (see the notes and paper above )
        #E = kornia.geometry.epipolar.find_essential(self.kpn_cur, self.kpn_ref)
        E, self.mask_match = cv2.findEssentialMat(self.kpn_cur, self.kpn_ref, focal=1, pp=(0., 0.), method=ransac_method, prob=0.999, threshold=0.0003)
        _, R, t, mask = cv2.recoverPose(E, self.kpn_cur, self.kpn_ref, focal=1, pp=(0., 0.))   
        
        
        self.stats.update('num_inliers',self.mask_match.sum())
        self.stats.update('num_outliers',self.mask_match.shape[0] - self.mask_match.sum())
        return R,t
    
    def estimatePoseTorch(self, kps_ref, kps_cur):
        kp_ref_u = self.cam.undistort_points(kps_ref)
        kp_cur_u = self.cam.undistort_points(kps_cur)
        kpn_ref = self.cam.unproject_points(kp_ref_u)
        kpn_cur = self.cam.unproject_points(kp_cur_u)
        kpn_ref = torch.tensor(kpn_ref).unsqueeze(0).float()
        kpn_cur = torch.tensor(kpn_cur).unsqueeze(0).float()
        kp_cur_u = torch.tensor(kp_cur_u).unsqueeze(0).float()
        kp_ref_u = torch.tensor(kp_ref_u).unsqueeze(0).float()
        
        kUseEssentialMatrixEstimation = True
        if kUseEssentialMatrixEstimation:
            # the essential matrix algorithm is more robust since it uses the five-point algorithm solver by D. Nister (see the notes and paper above)
            E = kornia.geometry.epipolar.find_essential(kpn_cur, kpn_ref)
        else:
            # just for the hell of testing fundamental matrix fitting ;-)
            F, self.mask_match = self.computeFundamentalMatrixTorch(kp_cur_u, kp_ref_u)
            E = self.cam.K.T @ F @ self.cam.K  # E = K.T * F * K
        K = torch.tensor(self.cam.K).unsqueeze(0).float()
        # self.removeOutliersFromMask(self.mask)  # do not remove outliers, the last unmatched/outlier features can be matched and recognized as inliers in subsequent frames
        R, t, _ = kornia.geometry.epipolar.motion_from_essential_choose_solution(E, K, K, kp_cur_u, kp_ref_u)
        return R.numpy()[0], t.numpy()[0]  # Rrc, trc (with respect to 'ref' frame)

    
    def drawFeatureTracks(self, img, reinit = False):
        if img.ndim == 2:
            draw_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            draw_img = img.copy()
        num_outliers = 0
        num_inliers = 0
        for i,pts in enumerate(zip(self.kps1, self.kps0)):
            if self.mask_match[i] == 0:
                num_outliers += 1
                continue
            p1, p2 = pts 
            a,b = p1.astype(int).ravel()
            c,d = p2.astype(int).ravel()
            cv2.line(draw_img, (a,b),(c,d), (0,255,0), 1)
            cv2.circle(draw_img,(a,b),1, (0,0,255),-1)  
            num_inliers += 1
        print("outliers:", num_outliers, "inliers:", num_inliers)
        return draw_img  
    
    def drawAllFeatureTracks(self, img, reinit = False):
        if img.ndim==2:
            draw_img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        else:
            draw_img = img.copy()           
   
        for i,pts in enumerate(zip(self.kps1, self.kps0)):

            p1, p2 = pts 
            a,b = p1.astype(int).ravel()
            c,d = p2.astype(int).ravel()
            cv2.line(draw_img, (a,b),(c,d), (0,255,0), 1)
            cv2.circle(draw_img,(a,b),1, (0,0,255),-1)   


        return draw_img  
