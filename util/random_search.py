import copy
import random
import json


def get_randomized_opts(opt):
    # TODO: build this out
    random_opts = []

    json_file = opt.random_search
    with open(json_file) as json_file:
        data = json.load(json_file)

        for perm in range(opt.max_random_permutations):
            new_opt = copy.deepcopy(opt)

            for opt_key in data:
                opt_range = data[opt_key]
                opt_val = random.choice(opt_range)
                setattr(new_opt, opt_key, opt_val)

            # # randomize the opts
            # if opt.random_lr_policy:
            #     lr_policies = ['linear', 'step', 'plateau', 'cosine']
            #     rand_policy = random.randint(0, len(lr_policies) - 1)
            #     new_opt.lr_policy = lr_policies[rand_policy]

            random_opts.append(new_opt)
    return random_opts
