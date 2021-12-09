import copy
import random

# TODO: build this out
def get_radomized_opts(opt):
    random_opts = []
    for perm in range(opt.max_random_permuations):
        new_opt = copy.deepcopy(opt)

        # randomize the opts
        if opt.random_lr_policy:
            lr_policies = ['linear', 'step', 'plateau', 'cosine']
            rand_policy = random.randint(0,len(lr_policies) - 1)
            new_opt.lr_policy = lr_policies[rand_policy]

        random_opts.append(new_opt)
    return random_opts