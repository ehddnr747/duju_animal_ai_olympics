import numpy as np




def state_1d_dim_calc(env):
    ob_spec = env.observation_spec()

    result = np.zeros((1,))

    for i,j in ob_spec.items():
        if len(j.shape) == 0:
            result = result + 1
        else:
            result = result + np.array(j.shape)

    return np.array(result,dtype=np.int)


def state_1d_flat(ob_dict):

    result = []

    for i, k in ob_dict.items():
        result.extend(list(np.reshape(k,[-1])))

    return np.array(result,dtype=np.float32)


# def dm_frame_preprocess(frame):
#     # Dim of frame [height, width, channel]
#
#     r_frame = np.moveaxis(frame, [0,1,2], [1,2,0])
#     r_frame = (r_frame / 128.0) - 1.0
#
#     # Return [channel, height, width]
#     return r_frame