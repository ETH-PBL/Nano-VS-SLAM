
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from utils.utils import save_json
import datetime
from utils.utils import set_seed
from visual_odometry.visual_odometry import VisualOdometry
from visual_odometry.camera import PinholeCamera
from visual_odometry.groundtruth import KittiVideoGroundTruth
from visual_odometry.dataset import VideoDataset, KittiDataset
default_config = {
    'im_h': 256,
    'im_w': 1024,
    'device': 'cpu',
    'plot': False,
    'plot_traj': False,
    'semantic_filter': False,
    'semantic_matching': False,
    'use_gt': False,
    'calculate_relative_error': False,
    'draw_all': False,


}
def parse_arguments():
    parser = argparse.ArgumentParser(description='Visual Odometry Evaluation')
    parser.add_argument('--calculate_relative_error', action='store_true', help='Calculate relative error')
    parser.add_argument('--method', type=str, default='kp2dtiny', help='Feature extraction method')
    parser.add_argument('--plot', action='store_true', help='Plot the 3D motion')
    parser.add_argument('--draw_all', action='store_true', help='Draw all feature tracks')
    parser.add_argument('--semantic_filter', action='store_true', help='Use semantic filter')
    parser.add_argument('--use_gt', action='store_true', help='Use ground truth')
    parser.add_argument('--semantic_matching', action='store_true', help='Use semantic matching')
    parser.add_argument('--video_path', type=str, default='/home/thomas/PycharmProjects/pyslam_ubuntu/pyslam/videos/kitti06/', help='Path to the video')
    parser.add_argument('--gt_path', type=str, default='06.txt', help='Path to the ground truth file')
    parser.add_argument('--video_name', type=str, default='video_color.mp4', help='Name of the video')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for processing')
    parser.add_argument('--weights_path', type=str, default='./checkpoints/tiny_A.ckpt', help='Path to the weights')
    parser.add_argument('--im_h', type=int, default=256, help='Image height')
    parser.add_argument('--im_w', type=int, default=1024, help='Image width')
    parser.add_argument('--use_light_glue', action='store_true', help='Use light glue')
    parser.add_argument('--plot_traj', action='store_true', help='Plot the trajectory')
    
    # Add more arguments here if needed
    args = parser.parse_args()
    return args

def kitty_params():
    fx = 718.856
    fy = 718.856
    cx = 607.1928
    cy =  185.2157
    return fx, fy, cx, cy

def ski_params(size):
    fx = 2317.6450198781713
    fy = 2317.6450198781713
    cx = size[1]/2.0
    cy = size[0]/2.0
    return fx, fy, cx, cy

class ErrorStatistics:
    def __init__(self, type= "absolute error"):
        self.type = type
        self.x_errors = []
        self.y_errors = []
        self.z_errors = []
        self.combined_errors = []

    def calculate_error(self, x, y, z, x_true, y_true, z_true):
        error = np.sqrt((x - x_true)**2 + (y - y_true)**2 + (z - z_true)**2)
        self.x_errors.append(abs(x - x_true))
        self.y_errors.append(abs(y - y_true))
        self.z_errors.append(abs(z - z_true))
        self.combined_errors.append(error)
        return error

    def get_mean(self):
        x_mean = np.mean(self.x_errors)
        y_mean = np.mean(self.y_errors)
        z_mean = np.mean(self.z_errors)
        combined_mean = np.mean(self.combined_errors)
        return x_mean, y_mean, z_mean, combined_mean

    def get_std(self):
        x_std = np.std(self.x_errors)
        y_std = np.std(self.y_errors)
        z_std = np.std(self.z_errors)
        combined_std = np.std(self.combined_errors)
        return x_std, y_std, z_std, combined_std

    def get_max(self):
        x_max = np.max(self.x_errors)
        y_max = np.max(self.y_errors)
        z_max = np.max(self.z_errors)
        combined_max = np.max(self.combined_errors)
        return x_max, y_max, z_max, combined_max

    def get_min(self):
        x_min = np.min(self.x_errors)
        y_min = np.min(self.y_errors)
        z_min = np.min(self.z_errors)
        combined_min = np.min(self.combined_errors)
        return x_min, y_min, z_min, combined_min
    def get_sum(self):
        x_sum = np.sum(self.x_errors)
        y_sum = np.sum(self.y_errors)
        z_sum = np.sum(self.z_errors)
        combined_sum = np.sum(self.combined_errors)
        return x_sum, y_sum, z_sum, combined_sum
    
    def get_error_stats(self):
        mean = self.get_mean()
        std = self.get_std()
        max = self.get_max()
        min = self.get_min()
        sum = self.get_sum()
        return {
            'type': self.type,
            'mean': mean,
            'std': std,
            'max': max,
            'min': min,
            'sum': sum
        }
        
    def print_error_stats(self):
        stats = self.get_error_stats()
        print("--- Error Statistics ---")
        print(f"Type: {stats['type']}")
        for key, value in stats.items():
            print(f"{key.capitalize()}: {value}")
def main(args):
    set_seed(42)
    artifacts = {"args": vars(args)}
    n_frames = 10000000
    #video_path = '/home/thomas/PycharmProjects/pyslam_ubuntu/pyslam/videos/ski_2.mp4'
    video_path = args.video_path + '/' + args.video_name

    #dataset = VideoDataset(args.video_path, args.video_name)
    dataset = KittiDataset(r"F:\Datasets\kitti\dataset", "09")
    if args.use_gt:
        gt = KittiVideoGroundTruth(args.video_path, args.gt_path)
        error_type = "relative error" if args.calculate_relative_error else "absolute error"
        err = ErrorStatistics(error_type)
    if args.plot:
        if args.plot_traj:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            #Plot the 3D positions
            ax.scatter(0, 0, 0, c='r', marker='o')
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.set_zlabel('Z [m]')
            ax.set_title('3D Plot of Motion')
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error loading video")
        return

    # Read the first frame
    frame = dataset.getImage(0)


    size = frame.shape
    
    #fx, fy, cx, cy = ski_params(size)
    cam_params = kitty_params()
    
    artifacts['camera_params'] = cam_params
    artifacts['video_size'] = size
    
    fx, fy, cx, cy = cam_params
    
    cam = PinholeCamera(size[1], size[0], fx, fy, cx, cy, [0, 0, 0, 0, 0])
    vo = VisualOdometry(cam, size, (args.im_h, args.im_w), {'device': args.device, "new_size": (args.im_h, args.im_w), 'semantic_filter': args.semantic_filter, 'semantic_matching': args.semantic_matching})


    artifacts['VisualOdometry'] = vo.get_info()
    
    if vo.gray:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        image = frame
        
    vo.init(image)
    
    # Create a 3D plot

    i_frame = 1
    scale = 1
    # Loop through the remaining frames
    while True:
        start_time = datetime.datetime.now()
        # Read the next frame
        frame = dataset.getImage(i_frame)

        if frame is None:
            break

        # if i_frame > 100:
        #     break
        if vo.gray:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            image = frame


        # Process the frame
        if args.use_gt:
            try:
                x_gt, y_gt, z_gt, scale = gt.getPoseAndAbsoluteScale(i_frame)
            except:
                print("Error getting ground truth")
                break
        
        t = vo.process_image(image, scale)
        
        if i_frame == 1:
            t_0 = t
            if args.use_gt:
                x_gt_0, y_gt_0, z_gt_0, scale_0 = x_gt, y_gt, z_gt, scale
        

        # Error calculation
        if args.use_gt and i_frame > 1:
            if args.calculate_relative_error:
                # applying the relative transformation to the last ground truth position
                _,_,_,absolute_scale = gt.getPoseAndAbsoluteScale(i_frame-1)
                t_last, rot_last = gt.extract_pose_values(i_frame-1)
                estimated_t = t_last + absolute_scale*rot_last.dot(vo.t_est[-1]).T
                estimated_R = rot_last.dot(vo.R_est[-1])
                t_curr, rot_curr = gt.extract_pose_values(i_frame )
                
                x, y, z = estimated_t[0,0], estimated_t[0,1], estimated_t[0,2]
                x_true, y_true, z_true = t_curr[0], t_curr[1], t_curr[2]
                t_error = np.sqrt(((t_curr - estimated_t)**2).sum())
                r, _ = cv2.Rodrigues(estimated_R.dot(rot_curr.T))
                r_error = np.linalg.norm(r)
            else:
                x, y, z = t[0]-t_0[0], t[1]-t_0[1], t[2]-t_0[2]
                x_true, y_true, z_true = x_gt-x_gt_0, y_gt-y_gt_0, z_gt-z_gt_0
            # Calculate Error
            error = err.calculate_error(x, y, z, x_true, y_true, z_true)
            print("Error: ", error)
        
        if args.plot:
            if args.plot_traj:
                #Plot the 3D positions
                new_data = np.array(t[0]- t_0[0]),\
                            np.array(t[1]- t_0[1]),\
                            np.array(t[2]- t_0[2])
                ax.scatter(*new_data, c='r', marker='o')
                if args.use_gt:
                    #Plot the ground truth
                    new_gt = np.array(x_gt-x_gt_0), np.array(y_gt - y_gt_0), np.array(z_gt - z_gt_0)
                    ax.scatter(*new_gt, c='b', marker='o')

            
                plt.pause(0.01)
            if hasattr(vo, 'mask_match') and not args.draw_all:
                img_debug = vo.drawFeatureTracks(image)
            else:
                img_debug = vo.drawAllFeatureTracks(image)
            end_time = datetime.datetime.now()
            elapsed_time = end_time - start_time
            start_time = end_time
            fps = 1./elapsed_time.total_seconds()
            cv2.putText(img_debug, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow('frame', img_debug)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        i_frame += 1
        if i_frame > n_frames:
            break
    artifacts['vo_stats'] = vo.stats.get_stats()
    if args.use_gt:
        err.print_error_stats()
        artifacts['error_stats'] = err.get_error_stats()
    artifacts['frames_processed'] = i_frame
    # Release the video capture object
    if args.plot:
        cap.release()
        if args.plot_traj:
            plt.show()
        #plt.show()
        cv2.destroyAllWindows()
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    save_json(artifacts, "./artifacts/vo_artifacts_" + args.method + '_' + timestamp + '.json')
    

if __name__ == "__main__":
    args = parse_arguments()
    main(args)