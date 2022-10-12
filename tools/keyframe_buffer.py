"""
Most of this is a modified version of code from the DeepVideoMVS repository 
at https://github.com/ardaduz/deep-video-mvs/blob/master/dvmvs/keyframe_buffer.py
"""

from collections import deque
import functools
import numpy as np



class DVMVS_Config:
    # train tuple settings
    train_minimum_pose_distance = 0.125
    train_maximum_pose_distance = 0.325
    train_crawl_step = 3

    # test tuple settings
    test_keyframe_buffer_size = 30
    test_keyframe_pose_distance = 0.1
    test_optimal_t_measure = 0.15
    test_optimal_R_measure = 0.0

def is_pose_available(pose):
    is_nan = np.isnan(pose).any()
    is_inf = np.isinf(pose).any()
    is_neg_inf = np.isneginf(pose).any()
    if is_nan or is_inf or is_neg_inf:
        return False
    else:
        return True

def is_valid_pair(
            reference_pose, 
            measurement_pose, 
            pose_dist_min, 
            pose_dist_max, 
            t_norm_threshold=0.05, 
            return_measure=False,
        ):
    combined_measure, _, t_measure = pose_distance(reference_pose, measurement_pose)

    if (pose_dist_min <= combined_measure <= pose_dist_max 
                                            and t_measure >= t_norm_threshold):
        result = True
    else:
        result = False

    if return_measure:
        return result, combined_measure
    else:
        return result

def pose_distance(reference_pose, measurement_pose):
    """
    :param reference_pose: 4x4 numpy array, reference frame camera-to-world pose 
        (not extrinsic matrix!)
    :param measurement_pose: 4x4 numpy array, measurement frame camera-to-world 
        pose (not extrinsic matrix!)
    :return combined_measure: float, combined pose distance measure
    :return R_measure: float, rotation distance measure
    :return t_measure: float, translation distance measure
    """
    rel_pose = np.dot(np.linalg.inv(reference_pose), measurement_pose)
    R = rel_pose[:3, :3]
    t = rel_pose[:3, 3]
    R_measure = np.sqrt(2 * (1 - min(3.0, np.matrix.trace(R)) / 3))
    t_measure = np.linalg.norm(t)
    combined_measure = np.sqrt(t_measure ** 2 + R_measure ** 2)
    return combined_measure, R_measure, t_measure

class KeyframeBuffer:
    def __init__(
            self, 
            buffer_size, 
            keyframe_pose_distance, 
            optimal_t_score, 
            optimal_R_score, 
            store_return_indices,
        ):
        self.buffer = deque([], maxlen=buffer_size)
        self.keyframe_pose_distance = keyframe_pose_distance
        self.optimal_t_score = optimal_t_score
        self.optimal_R_score = optimal_R_score
        self.__tracking_lost_counter = 0
        # mostly required for simulation of the frame selection
        self.__store_return_indices = store_return_indices  

    def calculate_penalty(self, t_score, R_score):
        degree = 2.0
        R_penalty = np.abs(R_score - self.optimal_R_score) ** degree
        t_diff = t_score - self.optimal_t_score
        if t_diff < 0.0:
            t_penalty = 5.0 * (np.abs(t_diff) ** degree)
        else:
            t_penalty = np.abs(t_diff) ** degree
        return R_penalty + t_penalty

    def try_new_keyframe(self, pose, image, dist_to_last_valid=None, index=None):
        if self.__store_return_indices and index is None:
            raise ValueError("Storing and returning the frame indices is "
                            f"requested in the constructor, but index=None is "
                            f"passed to the function")

        # In case valid frames are used, this helps guess if a gap in tracking 
        # when using valid frames and the indices are not indicative of time.
        if (dist_to_last_valid is not None and dist_to_last_valid > 30):
            self.buffer.clear()
            self.__tracking_lost_counter = 0
            if self.__store_return_indices:
                self.buffer.append((pose, image, index))
            else:
                self.buffer.append((pose, image))
            
            return 3
            
        if is_pose_available(pose):
            self.__tracking_lost_counter = 0
            if len(self.buffer) == 0:
                if self.__store_return_indices:
                    self.buffer.append((pose, image, index))
                else:
                    self.buffer.append((pose, image))
                # pose is available, new frame added but buffer was empty, this 
                # is the first frame, no depth map prediction will be done
                return 0  
            else:
                if self.__store_return_indices:
                    last_pose, last_image, last_index = self.buffer[-1]
                else:
                    last_pose, last_image = self.buffer[-1]

                combined_measure, R_measure, t_measure = pose_distance(pose, last_pose)

                if combined_measure >= self.keyframe_pose_distance:
                    if self.__store_return_indices:
                        self.buffer.append((pose, image, index))
                    else:
                        self.buffer.append((pose, image))
                    # pose is available, new frame added, everything is perfect, 
                    # and we will predict a depth map later
                    return 1  
                else:
                    # pose is available but not enough change has happened since 
                    # the last keyframe
                    return 2  
        else:
            self.__tracking_lost_counter += 1

            if self.__tracking_lost_counter > 30:
                if len(self.buffer) > 0:
                    self.buffer.clear()
                    # a pose reading has not arrived for over a second, tracking 
                    # is now lost
                    return 3  
                else:
                    return 4  # we are still very lost
            else:
                # pose is not available right now, but not enough time has 
                # passed to consider lost, there is still hope :)
                return 5  

    def get_best_measurement_frames(self, n_requested_measurement_frames):
        buffer_array = list(self.buffer)

        if self.__store_return_indices:
            reference_pose, reference_image, reference_index = buffer_array[-1]
        else:
            reference_pose, reference_image = buffer_array[-1]

        n_requested_measurement_frames = min(n_requested_measurement_frames,
                                                        len(buffer_array) - 1)

        penalties = []
        for i in range(len(buffer_array) - 1):
            measurement_pose = buffer_array[i][0]
            
            _, R_measure, t_measure = pose_distance(reference_pose, measurement_pose)
            penalty = self.calculate_penalty(t_measure, R_measure)
            penalties.append(penalty)
        indices = np.argpartition(penalties, n_requested_measurement_frames - 1)[:n_requested_measurement_frames]

        measurement_frames = []
        for index in indices:
            measurement_frames.append(buffer_array[index])
        return measurement_frames


class SimpleBuffer:
    def __init__(
            self,
            buffer_size,
            store_return_indices,
        ):
        self.buffer = deque([], maxlen=buffer_size + 1)
        self.__tracking_lost_counter = 0
        # mostly required for simulation of the frame selection
        self.__store_return_indices = store_return_indices  

    def try_new_keyframe(self, pose, image, index=None):
        if self.__store_return_indices and index is None:
            raise ValueError(f"Storing and returning the frame indices is "
                            f"requested in the constructor, but index=None is "
                            f"passed to the function")

        if is_pose_available(pose):
            self.__tracking_lost_counter = 0
            if len(self.buffer) == 0:
                if self.__store_return_indices:
                    self.buffer.append((pose, image, index))
                else:
                    self.buffer.append((pose, image))
                # pose is available, new frame added but buffer was empty, this 
                # is the first frame, no depth map prediction will be done
                return 0  
            else:
                if self.__store_return_indices:
                    self.buffer.append((pose, image, index))
                else:
                    self.buffer.append((pose, image))
                # pose is available, new frame added, everything is perfect, 
                # and we will predict a depth map later
                return 1  
        else:
            self.__tracking_lost_counter += 1

            if self.__tracking_lost_counter > 30:
                if len(self.buffer) > 0:
                    self.buffer.clear()
                    # a pose reading has not arrived for over a second, tracking 
                    # is now lost
                    return 2  
                else:
                    # we are still very lost
                    return 3  
            else:
                # pose is not available right now, but not enough time has 
                # passed to consider lost, there is still hope :)
                return 4  

    def get_measurement_frames(self):
        measurement_frames = list(self.buffer)[:-1]
        return measurement_frames

class OfflineKeyframeBuffer:
    def __init__(
            self,
            buffer_size,
            keyframe_pose_distance, 
            optimal_t_score,
            optimal_R_score,
            store_return_indices,
        ):
        self.buffer = deque([], maxlen=buffer_size)
        self.keyframe_pose_distance = keyframe_pose_distance
        self.optimal_t_score = optimal_t_score
        self.optimal_R_score = optimal_R_score
        self.__tracking_lost_counter = 0
        # mostly required for simulation of the frame selection
        self.__store_return_indices = store_return_indices  

    @functools.lru_cache()
    def calculate_penalty(self, t_score, R_score):
        degree = 2.0
        R_penalty = np.abs(R_score - self.optimal_R_score) ** degree
        t_diff = t_score - self.optimal_t_score
        if t_diff < 0.0:
            t_penalty = 5.0 * (np.abs(t_diff) ** degree)
        else:
            t_penalty = np.abs(t_diff) ** degree
        return R_penalty + t_penalty

    def try_new_keyframe(self, pose, image, index=None):
        if self.__store_return_indices and index is None:
            raise ValueError(f"Storing and returning the frame indices is "
                            f"requested in the constructor, but index=None is "
                            f"passed to the function")

        if is_pose_available(pose):
            self.__tracking_lost_counter = 0
            if len(self.buffer) == 0:
                if self.__store_return_indices:
                    self.buffer.append((pose, image, index))
                else:
                    self.buffer.append((pose, image))
                # pose is available, new frame added but buffer was empty, this 
                # is the first frame, no depth map prediction will be done
                return 0  
            else:
                if self.__store_return_indices:
                    last_pose, last_image, last_index = self.buffer[-1]
                else:
                    last_pose, last_image = self.buffer[-1]

                accept_frame = True
                for buffer_pose, _, _ in list(self.buffer):
                    combined_measure, _, _ = pose_distance(pose, buffer_pose)

                    if combined_measure < self.keyframe_pose_distance:
                        accept_frame = False
                        break

                if accept_frame:
                    if self.__store_return_indices:
                        self.buffer.append((pose, image, index))
                    else:
                        self.buffer.append((pose, image))
                    # pose is available, new frame added, everything is perfect, 
                    # and we will predict a depth map later
                    return 1  
                else:
                    # pose is available but not enough change has happened since 
                    # the last keyframe
                    return 2  
        else:
            self.__tracking_lost_counter += 1

            if self.__tracking_lost_counter > 30:
                if len(self.buffer) > 0:
                    self.buffer.clear()
                    # a pose reading has not arrived for over a second, tracking 
                    # is now lost
                    return 3  
                else:
                    # we are still very lost
                    return 4  
            else:
                # pose is not available right now, but not enough time has 
                # passed to consider lost, there is still hope :)
                return 5  

    def get_best_measurement_frames(self, n_requested_measurement_frames):
        buffer_array = list(self.buffer)

        if self.__store_return_indices:
            reference_pose, reference_image, reference_index = buffer_array[-1]
        else:
            reference_pose, reference_image = buffer_array[-1]

        n_requested_measurement_frames = min(n_requested_measurement_frames, 
                                                        len(buffer_array) - 1)

        penalties = []
        for i in range(len(buffer_array) - 1):
            measurement_pose = buffer_array[i][0]
            _, R_measure, t_measure = pose_distance(reference_pose, measurement_pose)

            penalty = self.calculate_penalty(t_measure, R_measure)
            penalties.append(penalty)
        indices = np.argpartition(penalties, n_requested_measurement_frames - 1)[:n_requested_measurement_frames]

        measurement_frames = []
        for index in indices:
            measurement_frames.append(buffer_array[index])
        return measurement_frames

    def get_best_measurement_frames_for_0index(self, n_requested_measurement_frames):
        buffer_array = list(self.buffer)[1:]

        if len(buffer_array) == 0:
            return []

        if self.__store_return_indices:
            reference_pose, _, _ = buffer_array[0]
        else:
            reference_pose, _ = buffer_array[0]

        n_requested_measurement_frames = min(n_requested_measurement_frames, len(buffer_array) - 1)

        penalties = []
        for i in range(len(buffer_array)):
            measurement_pose = buffer_array[i][0]
            _, R_measure, t_measure = pose_distance(reference_pose, measurement_pose)
            penalty = self.calculate_penalty(t_measure, R_measure)
            penalties.append(penalty)
        
        indices = np.argpartition(penalties, n_requested_measurement_frames - 1)[:n_requested_measurement_frames]
        measurement_frames = []
        for index in indices:
            measurement_frames.append(buffer_array[index])
        return measurement_frames