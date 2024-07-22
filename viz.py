import os
import cv2
import numpy as np

image_dir = "D:/Rope3D_data"
output_dir = "F:/rope3d_eval/data"

output_lists = os.listdir(output_dir)
data_names = [name.split('.')[0] for name in output_lists]
data_names.sort()

for data_name in data_names:
    image_path = os.path.join(image_dir, 'image_2', data_name + ".jpg")
    image = cv2.imread(image_path)
    
    #read label (KITTI format)
    label_path = os.path.join(output_dir, data_name + ".txt")
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    #read calibration
    calib_path = os.path.join(image_dir, 'calib', data_name + ".txt")
    with open(calib_path, 'r') as f:
        calib = f.readlines()
    
    P = calib[0].split(' ')[1:]
    P = np.array([float(x) for x in P]).reshape(3, 4)
        
    #read denorm
    denorm_path = os.path.join(image_dir, 'denorm', data_name + ".txt")
    with open(denorm_path, 'r') as f:
        denorm = f.readlines()
        
    dnorm = denorm[0].split(' ')
    d0, d1, d2, d3 = float(dnorm[0]), float(dnorm[1]), float(dnorm[2]), float(dnorm[3])
    rot_d = np.array([
        [1, 0, 0],
        [0, -d1, +d2],
        [0, -d2, -d1]
    ])
        
    for line in lines:
        line = line[:-1].split(' ')
        object_type = line[0]
        object_score = float(line[-1])
        
        #draw 2d box
        bbox = np.array([float(x) for x in line[4:8]])
        bbox = bbox.astype(np.int32)
        
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        obj_text = object_type + " " + str(object_score)
        cv2.putText(image, obj_text, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        #draw 3d box
        obj_h, obj_w, obj_l = float(line[8]), float(line[9]), float(line[10])
        obj_x, obj_y, obj_z = float(line[11]), float(line[12]), float(line[13])
        obj_rot_y = float(line[14])
        
        obj_ext = np.array([
            [np.cos(obj_rot_y), 0, np.sin(obj_rot_y), obj_x],
            [0, 1, 0, obj_y],
            [-np.sin(obj_rot_y), 0, np.cos(obj_rot_y), obj_z],
            [0, 0, 0, 1]
        ])
        
        obj_ext[:3, :3] = rot_d @ obj_ext[:3, :3]
        
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        
        points = np.array([
            [obj_l, obj_l, obj_l, obj_l, -obj_l, -obj_l, -obj_l, -obj_l],
            [-obj_h, -obj_h, obj_h, obj_h, -obj_h, -obj_h, obj_h, obj_h],
            [-obj_w, obj_w, obj_w, -obj_w, -obj_w, obj_w, obj_w, -obj_w]
        ])
        points[0, :] = points[0, :] / 2
        points[1, :] = (points[1, :] - obj_h) / 2
        points[2, :] = points[2, :] / 2
        
        points = np.vstack([points, np.ones(points.shape[1])])
        points = np.dot(obj_ext, points)
        points_2d = np.dot(P, points)
        
        for edge in edges:
            cv2.line(image, (int(points_2d[0, edge[0]]/points_2d[2, edge[0]]), int(points_2d[1, edge[0]]/points_2d[2, edge[0]])), 
                            (int(points_2d[0, edge[1]]/points_2d[2, edge[1]]), int(points_2d[1, edge[1]]/points_2d[2, edge[1]])), (0, 0, 255), 2)
        
        
        
        
    imh, imw, _ = image.shape
    image = cv2.resize(image, (int(imw/2), int(imh/2)))
    cv2.imshow("image", image)
    cv2.waitKey(0)