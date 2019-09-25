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