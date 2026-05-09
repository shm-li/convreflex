import math

def translate_to_nonscaled_shortcut_params(pos_per_channel, 
                           threshold_per_channel, 
                           type_per_channel,
                           n_channels,
                           effective_scale,
                           output_offset) -> list:
    # if len(pos_per_channel) == 0:
    #     return [], []
    # all_empty = True
    # for each in pos_per_channel:
    #     if each != []:
    #         all_empty = False
    #         break
    # if all_empty:
    #     return [], []

    if len(pos_per_channel) != n_channels:
        raise RuntimeError("The bpoint pos file's channel {:d} does not match {:d}".format(len(pos_per_channel), n_channels))
    bounds = []
    comp_steps = []

    for channel in range(n_channels):
        acc_bounds = []
        pos = pos_per_channel[channel]
        if len(pos) > 1: raise RuntimeError("No support for >1 checks")
        for p_idx, p in enumerate(pos):
            # print("TESTING", channel, p_idx, pos, type_per_channel[channel], threshold_per_channel[channel])
            check_type = type_per_channel[channel][p_idx]
            threshold = threshold_per_channel[channel][p_idx]
            scale = effective_scale[channel]
            offset = output_offset

            if check_type == 1: # check if intermediate result is too low
                requantized_bound = int(math.floor(
                        (threshold - offset) / scale))
                while ((requantized_bound) * scale + offset \
                            >= (threshold - 0.5)):
                    requantized_bound -= 1
                assert (requantized_bound + 1) * scale + offset \
                            >= threshold - 0.5
            elif check_type == 2: # check if intermediate result is too big
                # check_type == 2 (overflow) is very rare. 
                #   We ignore this case,
                #   so we don't spend an else branch on something that
                #   nearly never happens
                check_type = 0
                requantized_bound = 0
                # requantized_bound = int(math.ceil(
                #         (threshold - offset) / scale))
                # while ((requantized_bound) * scale + offset \
                #             <= (threshold + 0.5)):
                #     requantized_bound += 1
                # assert (requantized_bound - 1) * scale + offset \
                #             <= threshold + 0.5
            else:
                requantized_bound = 0

            acc_bounds.extend([check_type, requantized_bound])
        step_counter = [len(pos)]
        step_counter.extend([p for p in pos])

        bounds.extend(acc_bounds)
        comp_steps.extend(step_counter)

    return comp_steps, bounds #, min_fluc, max_fluc