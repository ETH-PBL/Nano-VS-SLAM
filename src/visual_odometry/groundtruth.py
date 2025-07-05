import numpy as np
import sys


class GroundTruth(object):
    def __init__(self, path, name, associations=None, type=None):
        self.path = path
        self.name = name
        self.type = type
        self.associations = associations
        self.filename = None
        self.file_associations = None
        self.data = None
        self.scale = 1

    def getDataLine(self, frame_id):
        return self.data[frame_id].strip().split()

    def getPoseAndAbsoluteScale(self, frame_id):
        return 0, 0, 0, 1

    # convert the dataset into 'Simple' format  [x,y,z,scale]
    def convertToSimpleXYZ(self, filename="groundtruth.txt"):
        out_file = open(filename, "w")
        num_lines = len(self.data)
        print("num_lines:", num_lines)
        for ii in range(num_lines):
            x, y, z, scale = self.getPoseAndAbsoluteScale(ii)
            if ii == 0:
                scale = 1  # first sample: we do not have a relative
            out_file.write("%f %f %f %f \n" % (x, y, z, scale))
        out_file.close()


class KittiVideoGroundTruth(GroundTruth):
    def __init__(self, path, name, associations=None, type="kitti"):
        super().__init__(path, name, associations, type)
        self.scale = 1
        self.filename = path + "/" + name
        with open(self.filename) as f:
            self.data = f.readlines()
            self.found = True
        if self.data is None:
            sys.exit(
                "ERROR while reading groundtruth file: please, check how you deployed the files and if the code is consistent with this!"
            )

    def getPoseAndAbsoluteScale(self, frame_id):
        ss = self.getDataLine(frame_id - 1)
        x_prev = self.scale * float(ss[3])
        y_prev = self.scale * float(ss[7])
        z_prev = self.scale * float(ss[11])
        ss = self.getDataLine(frame_id)
        x = self.scale * float(ss[3])
        y = self.scale * float(ss[7])
        z = self.scale * float(ss[11])
        abs_scale = np.sqrt(
            (x - x_prev) * (x - x_prev)
            + (y - y_prev) * (y - y_prev)
            + (z - z_prev) * (z - z_prev)
        )
        return x, y, z, abs_scale

    def extract_pose_values(self, frame_id):
        pose_values = self.getDataLine(frame_id)
        # convert all values to floats
        pose_values = [float(value) for value in pose_values]
        # Reshape the pose values into a 3x4 matrix
        pose_matrix = np.reshape(pose_values, (3, 4))

        # Ensure self.scale is a float
        scale = float(self.scale)
        # Ensure pose_matrix[:, 3] is a numpy array of floats
        pose_matrix[:, 3] = pose_matrix[:, 3]

        # Now you can perform the multiplication
        translation = pose_matrix[:, 3]
        translation = translation * scale

        # Extract rotation matrix
        rotation_matrix = pose_matrix[:, :3]
        return translation, rotation_matrix

    def getAbsoluteScale(self, frame_id):
        self.trueX = 0
        self.trueY = 0
        self.trueZ = 0
        return 1
