import time
import pickle
import re
import sys


# redundancy omitting with clamping prediction {
class StepStats:
    def __init__(self):
        self.total_sample = 0
        self.total_under_clamp_sample = 0
        self.total_over_clamp_sample = 0
        self.val_freq = {}
        self.val_under_clamp_count = {}
        self.val_over_clamp_count = {}
        self.val_wrong_under_clamp_pred_error_sum = {}
        self.val_wrong_over_clamp_pred_error_sum = {}
        # self.val_freq_when_clamping = {}

        self.smaller_than_val_freq = None
        self.larger_than_val_freq = None
        self.smaller_than_val_clamp_under_freq = None
        # self.smaller_than_val_clamp_over_freq = None
        # self.larger_than_val_clamp_under_freq = None
        self.larger_than_val_clamp_over_freq = None
        self.smaller_than_val_wrong_under_clamp_pred_error = None
        self.larger_than_val_wrong_over_clamp_pred_error = None
        self.val_min = None
        self.val_max = None
    

    def add_val(self, clamp_type, output, val):
        self.total_sample += 1
        self.__add_or_create(self.val_freq, val)
        if clamp_type == 1:
            self.total_under_clamp_sample += 1
            self.__add_or_create(self.val_under_clamp_count, val)
        elif clamp_type == 2:
            self.total_over_clamp_sample += 1
            self.__add_or_create(self.val_over_clamp_count, val)
        elif clamp_type != 0:
            assert (output != -128) and (output != 127)
            self.__inc_or_create(self.val_wrong_under_clamp_pred_error_sum, 
                                 val, abs(-128 - output))
            self.__inc_or_create(self.val_wrong_over_clamp_pred_error_sum, 
                                 val, abs(127 - output))
        #     self.total_clamp_sample += 1
        #     self.__add_or_create(self.val_freq_when_clamping, val)

    
    # Probability to over-clamp at current step when the accumulated val is
    #   greater than n
    def get_clamp_prob_when_val_over(self, n):
        if (n in self.larger_than_val_freq) \
                and (n not in self.larger_than_val_clamp_over_freq):
            assert self.larger_than_val_freq[n] != 0
            return 0
        return self.larger_than_val_clamp_over_freq[n] \
                    / self.larger_than_val_freq[n]

    
    # Probability to under-clamp at current step when the accumulated val is
    #   less than n
    def get_clamp_prob_when_val_under(self, n):
        if (n in self.smaller_than_val_freq) \
                and (n not in self.smaller_than_val_clamp_under_freq):
            assert self.smaller_than_val_freq[n] != 0
            return 0
        return self.smaller_than_val_clamp_under_freq[n] \
                    / self.smaller_than_val_freq[n]
    
    
    def get_larger_than_val_freq(self, n):
        return self.larger_than_val_freq[n]


    def get_larger_than_val_clamp_over_freq(self, n):
        return self.larger_than_val_clamp_over_freq[n]
    

    def get_smaller_than_val_freq(self, n):
        return self.smaller_than_val_freq[n]
    

    def get_smaller_than_val_clamp_under_freq(self, n):
        return self.smaller_than_val_clamp_under_freq[n]
    
    
    def get_wrong_pred_error_when_val_over(self, n):
        if (n in self.larger_than_val_freq) \
            and (n not in self.larger_than_val_wrong_over_clamp_pred_error):
            assert self.larger_than_val_freq[n] != 0
            return 0
        return self.larger_than_val_wrong_over_clamp_pred_error[n] \
                    / self.larger_than_val_freq[n]
    

    def get_wrong_pred_error_when_val_under(self, n):
        if (n in self.smaller_than_val_freq) \
            and (n not in self.smaller_than_val_wrong_under_clamp_pred_error):
            assert self.smaller_than_val_freq[n] != 0
            return 0
        return self.smaller_than_val_wrong_under_clamp_pred_error[n] \
                    / self.smaller_than_val_freq[n]
    

    def get_val_to_over_clamp_with_confidence(self, target_conf: float):
        if self.larger_than_val_freq == None:
            self.smaller_than_val_freq, self.larger_than_val_freq, \
                        self.val_min, self.val_max \
                    = self.__gen_val_cumulative_freq(self.val_freq)
            # self.smaller_than_val_clamp_over_freq, \
            _, self.larger_than_val_clamp_over_freq, _, _ \
                    = self.__gen_val_cumulative_freq(self.val_over_clamp_count)
        if self.total_over_clamp_sample == 0: return None, None
        best_val = None
        best_freq = 0
        if target_conf <= 1.00:
            for val in range(self.val_min, self.val_max): # skip val_max 
                                                            # cus it's 0
                if val not in self.val_freq: continue
                if self.get_clamp_prob_when_val_over(val) >= target_conf: 
                    if self.larger_than_val_freq[val] > best_freq:
                        best_freq = self.larger_than_val_freq[val]
                        best_val = val
        elif target_conf > 1.00: # find a n with a margin
            margin = target_conf - 1.00
            # Find the first val_100_conf to have 100% clamping probability
            for val in range(self.val_min, self.val_max): # skip val_max
                if val not in self.val_freq: continue
                if self.get_clamp_prob_when_val_over(val) >= 1.00: 
                    val_100_conf = val
                    freq_100_conf = self.larger_than_val_freq[val]
                    break # find the first val that satisfies 100% conf
            else: # No val to satisfy 100% conf
                return None, None
            # Skip a percentage of samples (margin) that satisfies 100%
            for val in range(val_100_conf, self.val_max):
                if val not in self.val_freq: continue
                if self.get_clamp_prob_when_val_over(val) >= 1.00: 
                    if self.larger_than_val_clamp_over_freq[val] \
                            * (1 + margin) > freq_100_conf:
                            continue
                    if self.larger_than_val_freq[val] > best_freq:
                        best_freq = self.larger_than_val_freq[val]
                        best_val = val
        if best_val == None:
           return None, None
        return best_val, best_freq / self.total_sample

    
    def get_val_to_under_clamp_with_confidence(self, target_conf: float):
        if self.smaller_than_val_freq == None:
            self.smaller_than_val_freq, self.larger_than_val_freq, \
                        self.val_min, self.val_max \
                    = self.__gen_val_cumulative_freq(self.val_freq)
            # self.smaller_than_val_clamp_under_freq, \
            #             self.larger_than_val_clamp_under_freq, _, _ \
            self.smaller_than_val_clamp_under_freq, _, _, _ \
                    = self.__gen_val_cumulative_freq(self.val_under_clamp_count)
        # print("ÄA", self.smaller_than_val_freq, self.val_min, self.val_max)
        # print("ÄA", self.larger_than_val_freq, self.val_min, self.val_max)
        if self.total_under_clamp_sample == 0: return None, None
        best_val = None
        best_freq = 0
        if target_conf <= 1.00:
            for val in range(self.val_max, self.val_min, -1): # skip val_min-1
                                                                # cuz it's 0
                if val not in self.val_freq: continue
                if self.get_clamp_prob_when_val_under(val) >= target_conf: 
                    # print("FREQ OF THIS:", self.smaller_than_val_freq[k])
                    # print("PROB OF CLAMP:", self.get_clamp_prob_when_val_under_n(k))
                    if self.smaller_than_val_freq[val] > best_freq:
                        best_freq = self.smaller_than_val_freq[val]
                        best_val = val
                        # print("SUBSTITUTING best", best_freq, best_k)
        elif target_conf > 1.00: # find a n with a margin
            margin = target_conf - 1.00
            # Find the first val_100_conf to have 100% clamping probability
            for val in range(self.val_max, self.val_min, -1): # skip val_min-1
                if val not in self.val_freq: continue
                if self.get_clamp_prob_when_val_under(val) >= 1.00: 
                    val_100_conf = val
                    freq_100_conf = self.smaller_than_val_freq[val]
                    break # find the first val that satisfies 100% conf
            else: # No val to satisfy 100% conf
                return None, None
            # Skip a percentage of samples (margin) that satisfies 100%
            for val in range(val_100_conf, self.val_min, -1):
                if val not in self.val_freq: continue
                if self.get_clamp_prob_when_val_under(val) >= 1.00: 
                    if self.smaller_than_val_clamp_under_freq[val] \
                            * (1 + margin) > freq_100_conf:
                            continue
                    if self.smaller_than_val_freq[val] > best_freq:
                        best_freq = self.smaller_than_val_freq[val]
                        best_val = val

        if best_val == None:
           return None, None
        return best_val, best_freq / self.total_sample
    

#TODO: Return three values?
#TODO: Test these
    def get_val_to_over_clamp_with_confidence_and_error_bound(
            self, 
            target_conf: float,
            target_error: int # absolute error
    ):
        if self.larger_than_val_freq == None:
            self.smaller_than_val_freq, self.larger_than_val_freq, \
                        self.val_min, self.val_max \
                    = self.__gen_val_cumulative_freq(self.val_freq)
            # self.smaller_than_val_clamp_over_freq, \
            _, self.larger_than_val_clamp_over_freq, _, _ \
                    = self.__gen_val_cumulative_freq(self.val_over_clamp_count)
            _, self.larger_than_val_wrong_over_clamp_pred_error, _, _ \
                    = self.__gen_val_cumulative_freq(
                        self.val_wrong_over_clamp_pred_error_sum)
        if self.total_over_clamp_sample == 0: return None, None
        best_val = None
        best_freq = 0
        for val in range(self.val_min, self.val_max): # skip val_max cus it's 0
            if val not in self.val_freq: continue
            if self.get_clamp_prob_when_val_over(val) >= target_conf and \
                self.get_wrong_pred_error_when_val_over(val) <= target_error: 
                if self.larger_than_val_freq[val] > best_freq:
                    best_freq = self.larger_than_val_freq[val]
                    best_val = val
        if best_val == None:
           return None, None
        return best_val, best_freq / self.total_sample

    
    def get_val_to_under_clamp_with_confidence_and_error_bound(
        self, 
        target_conf: float,
        target_error: int # absolute error
    ):
        if self.smaller_than_val_freq == None:
            self.smaller_than_val_freq, self.larger_than_val_freq, \
                        self.val_min, self.val_max \
                    = self.__gen_val_cumulative_freq(self.val_freq)
            # self.smaller_than_val_clamp_under_freq, \
            #             self.larger_than_val_clamp_under_freq, _, _ \
            self.smaller_than_val_clamp_under_freq, _, _, _ \
                    = self.__gen_val_cumulative_freq(self.val_under_clamp_count)
            self.smaller_than_val_wrong_under_clamp_pred_error, _, _, _ \
                    = self.__gen_val_cumulative_freq(
                        self.val_wrong_under_clamp_pred_error_sum)
        # print("ÄA", self.smaller_than_val_freq, self.val_min, self.val_max)
        # print("ÄA", self.larger_than_val_freq, self.val_min, self.val_max)
        if self.total_under_clamp_sample == 0: return None, None
        best_val = None
        best_freq = 0
        for val in range(self.val_max, self.val_min, -1): # skip val_min - 1
                                                                # cuz it's 0
            if val not in self.val_freq: continue
            if self.get_clamp_prob_when_val_under(val) >= target_conf and \
                self.get_wrong_pred_error_when_val_under(val) <= target_error: 
                # print("FREQ OF THIS:", self.smaller_than_val_freq[k])
                # print("PROB OF CLAMP:", self.get_clamp_prob_when_val_under_n(k))
                if self.smaller_than_val_freq[val] > best_freq:
                    best_freq = self.smaller_than_val_freq[val]
                    best_val = val
                    # print("SUBSTITUTING best", best_freq, best_k)
        if best_val == None:
           return None, None
        return best_val, best_freq / self.total_sample


    def __gen_val_cumulative_freq(self, freq_dict):
        if freq_dict == {}:
            return None, None, 0, 0
        res = {}
        keys = list(sorted(freq_dict.keys()))
        accum = 0
        for k in keys:
            res[k] = accum
            v = freq_dict[k]
            accum += v
        res_reverse = {}
        keys_reverse = list(sorted(freq_dict.keys(), reverse=True))
        accum = 0
        for k in keys_reverse:
            res_reverse[k] = accum
            v = freq_dict[k]
            accum += v
        return res, res_reverse, keys[0], keys[-1]


    def __add_or_create(self, dist, val):
        if val in dist: dist[val] += 1
        else: dist[val] = 1
    

    def __inc_or_create(self, dist, val, inc):
        if val in dist: dist[val] += inc
        else: dist[val] = inc
# }


class ChannelStats:
    def __init__(self):
        self.per_neuron_steps = 0 # once set, this should be constant
        self.per_neuron_safely_omittable_step = None
        self.profiled_steps = 0
        self.terminated_steps = 0
        self.effectless_steps = 0
        self.safely_omittable_steps = 0

        self.underflow_clamping_count = 0
        self.overflow_clamping_count = 0
        self.no_termination_count = 0
        self.underflow_termination_count = 0
        self.overflow_termination_count = 0
        self.no_safely_omittable_count = 0
        self.underflow_omittable_count = 0
        self.overflow_omittable_count = 0
        self.neuron_count = 0
        self.clamped_neuron_count = 0

        # Stats
        self.per_step_termination_probability = None

# redundancy omitting with clamping prediction {
        self.steps = None
        self.per_step_predictive_termination_probability_non_accum = None
        self.per_step_predictive_termination_threshold = []
        self.per_step_predictive_termination_type = [] # 0None, 1under, 2over
# }
    
    def add_profiling_data(self, termination_type, clamping_type,
                           total, terminated, effectless, safely_omittable):
        if self.per_neuron_steps == 0:
            self.per_neuron_steps = total
            self.per_neuron_safely_omittable_step = [0 for i in range(total + 1)]
        elif self.per_neuron_steps != total: raise RuntimeError("Shape change")

        self.profiled_steps += total
        self.terminated_steps += terminated
        self.effectless_steps += effectless
        self.safely_omittable_steps += safely_omittable
        self.per_neuron_safely_omittable_step[total - safely_omittable] += 1

        if termination_type == "None":
            self.no_termination_count += 1
        elif termination_type == "Underflow":
            self.underflow_termination_count += 1
        elif termination_type == "Overflow":
            self.overflow_termination_count += 1
        else: raise RuntimeError("Unrecognized termination type")

        if (clamping_type == "None") or (safely_omittable == 0):
            self.no_safely_omittable_count += 1
        elif (clamping_type == "Underflow") and (safely_omittable != 0):
            self.underflow_omittable_count += 1
        elif (clamping_type == "Overflow") and (safely_omittable != 0):
            self.overflow_omittable_count += 1

        self.neuron_count += 1
        if clamping_type != "None":
            self.clamped_neuron_count += 1
            if clamping_type == "Underflow":
                self.underflow_clamping_count += 1
            elif clamping_type == "Overflow":
                self.overflow_clamping_count += 1
        if termination_type != "None":
            if termination_type != clamping_type:
                # Here it's not raising error, cuz dynamic bounds can cause it
                # TODO: correctly reflect dynamic bounds being enabled
                #  in profiling data, and handle the special case here
                print("Warning: Unmatching type of clamping")
                #raise RuntimeError("Unmatching type of clamping")
        # print("ADDED", termination_type, clamping_type, total, terminated, effectless, safely_omittable)
        

# redundancy omitting with clamping prediction {
    def add_intermediate_accumulation_trace(self, clamp_type, output, trace):
        if len(trace) != self.per_neuron_steps + 1:
            raise RuntimeError("Unmatching trace length")
        if self.steps == None:
            self.steps \
                    = [StepStats() for i in range(self.per_neuron_steps + 1)]
        for i in range(len(trace)):
            # Count completed steps
            self.steps[i].add_val(clamp_type, output, trace[i])
# }


    # According to the profiled frequency of safe termination at each step,
    #   select n best positions to make termination checks
    def gen_best_check_pos(self, n=2):
        # Create a cumulative probability distribution function
        self.__gen_termination_probability()
        prob_func = lambda x : self.per_step_termination_probability[x]
        num_steps = self.per_neuron_steps

        if n > num_steps: raise RuntimeError("Too many checks")
        if n > 2: raise RuntimeError("Up to 2 checks supported by runtime lib")
        
        current_best_candidates, best_o = \
            self.__find_n_points_to_maximize_omission(n, num_steps, prob_func)


        self.best_check_pos = current_best_candidates
        self.best_check_omission = best_o
        # print("DIST:", self.per_step_termination_probability)
        # print("BEST POS:", self.best_check_pos, self.best_check_omission)
    

    # Only 1 check supported!!
    # TODO: figure out a way to estimate omission for checks >= 2.
    #       How does the first check's success rate affect checks behind it?
    #       We probably need another round of profiling.
    def gen_best_predictive_check_pos(self, clamping_prediction_confidence):
        # Create a cumulative probability distribution function
        self.__gen_termination_probablility_with_predicted_clamping(
                clamping_prediction_confidence)
        num_steps = self.per_neuron_steps
        best_omit = 0
        best_step = None
        for i in range(1, num_steps + 1):
            term_type = self.per_step_predictive_termination_type[i]
            if term_type == None:
                continue
            prob = self.\
                    per_step_predictive_termination_probability_non_accum[i]
            steps_left = num_steps - i
            omit = prob * steps_left
            if omit > best_omit:
                best_step = i
                best_omit = omit
        if best_omit == 0: # No termination
            self.best_predictive_check_pos = []
        else:
            self.best_predictive_check_pos = [best_step]
        self.best_predictive_check_omission = best_omit
    

    # For displaying dist
    def get_dist(self):
        self._gen_termination_probability()
        return(self.per_step_termination_probability)
    

    def __find_n_points_to_maximize_omission(self, n: int, 
                                             num_steps: int, prob_func):
        def step_omission(stop_steps, num_steps, stop_prob_func):
            f = stop_prob_func
            o = (num_steps - stop_steps[0]) * f(stop_steps[0])
            for i in range(1, len(stop_steps)):
                o += (num_steps - stop_steps[i]) \
                        * (f(stop_steps[i]) - f(stop_steps[i - 1]))
            return o
        
        candidates = [0 + i for i in range(n)]
        loop_depth = n - 1
        current_best_candidates = [num_steps for each in candidates]
        best_o = 0
        while True:
            if loop_depth == n - 1:
                if loop_depth != 0:
                    candidates[loop_depth] = candidates[loop_depth - 1] + 1
                while candidates[loop_depth] < num_steps:
                    o = step_omission(candidates, num_steps, prob_func)
                    if o > best_o:
                        best_o = o; 
                        current_best_candidates = [each for each in candidates]
                    candidates[loop_depth] += 1
                loop_depth -= 1
            else:
                if loop_depth < 0: break

                if candidates[loop_depth] < num_steps:
                    candidates[loop_depth] += 1
                    loop_depth += 1
                    candidates[loop_depth] = candidates[loop_depth - 1]
                else:
                    loop_depth -= 1
        if best_o == 0: # No termination
            current_best_candidates = []
        
        return current_best_candidates, best_o


    def __gen_termination_probability(self):
        accum_term_prob = [0 for i in \
                           range(len(self.per_neuron_safely_omittable_step))] 
        for i, freq in enumerate(self.per_neuron_safely_omittable_step):
            for j in range(i, len(self.per_neuron_safely_omittable_step)):
                accum_term_prob[j] += freq
        for i in range(len(accum_term_prob)):
            accum_term_prob[i] = accum_term_prob[i] / accum_term_prob[-1]
        self.per_step_termination_probability = accum_term_prob


# redundancy omitting with clamping prediction {
    def __gen_termination_probablility_with_predicted_clamping(self, conf):
        term_prob = []#[0] # First step always 0 # No, self.steps has step 0!
        term_threshold = []#[None]
        term_type = []#[None]
        for step in self.steps:
            underflow_val, underflow_prob = \
                step.get_val_to_under_clamp_with_confidence(conf)
            overflow_val, overflow_prob = \
                step.get_val_to_under_clamp_with_confidence(conf)
            if underflow_val is None and overflow_prob is None:
                term_threshold.append(None)
                term_prob.append(0)
                term_type.append(0) # No termination
            elif overflow_val is None:
                term_threshold.append(underflow_val)
                term_prob.append(underflow_prob)
                term_type.append(1) # Underflow
            elif underflow_val is None:
                term_threshold.append(overflow_val)
                term_prob.append(overflow_prob)
                term_type.append(2) # Overflow
            else:
                if underflow_prob >= overflow_prob:
                    term_threshold.append(underflow_val)
                    term_prob.append(underflow_prob)
                    term_type.append(1) # Underflow
                else:
                    term_threshold.append(overflow_val)
                    term_prob.append(overflow_prob)
                    term_type.append(2) # Overflow
        self.per_step_predictive_termination_probability_non_accum = term_prob
        self.per_step_predictive_termination_type = term_type
        self.per_step_predictive_termination_threshold = term_threshold
# }   


class LayerStats:
    def __init__(self, idx, type):
        self.idx = idx
        self.type = type
        self.channels = []
    
    def add_profiling_data(self, ch, termination_type, clamping_type, total,
                           terminated, effectless, safely_omittable):
        if ch > len(self.channels) - 1:
            self.channels.append(ChannelStats())
            # raise RuntimeError("Non-existing channel. "
            #                    "Make sure ch number is consecutive!")
        # print("ADDING TO CHANNEL", ch)
        self.channels[ch].add_profiling_data(termination_type, clamping_type,
                                             total, terminated, effectless,
                                             safely_omittable)
        
# redundancy omitting with clamping prediction {
    def add_trace(self, ch, clamp_type, output, trace):
        if ch > len(self.channels) - 1:
            raise RuntimeError("Non-existing channel. "
                               "Make sure trace is always printed after"
                               "channel data!")
        self.channels[ch].add_intermediate_accumulation_trace(clamp_type, 
                                                              output, trace)
# }


    def get_dist(self):
        ret = []
        for ch in self.channels:
            ret.append(ch.get_dist())
        return ret


    def get_total_steps(self):
        ret = 0
        for ch_info in self.channels:
            ret += ch_info.profiled_steps
        total = sum(ch.profiled_steps for ch in self.channels)
        assert total == ret
        return total

    
    def get_terminated_steps(self):
        total = sum(ch.terminated_steps for ch in self.channels)
        return total
    

    def get_effectless_steps(self):
        total = sum(ch.effectless_steps for ch in self.channels)
        return total


    def get_safely_omittable_steps(self):
        total = sum(ch.safely_omittable_steps for ch in self.channels)
        return total
    

    def get_neuron_count(self):
        total = sum(ch.neuron_count for ch in self.channels)
        return total
    

    def get_clamped_neuron_count(self):
        total = sum(ch.clamped_neuron_count for ch in self.channels)
        return total


    def get_underflow_termination_count(self):
        total = sum(ch.underflow_termination_count for ch in self.channels)
        return total
    

    def get_overflow_termination_count(self):
        total = sum(ch.overflow_termination_count for ch in self.channels)
        return total


    def get_underflow_clamping_count(self):
        total = sum(ch.underflow_clamping_count for ch in self.channels)
        return total
    

    def get_overflow_clamping_count(self):
        total = sum(ch.overflow_clamping_count for ch in self.channels)
        return total


    def get_no_safely_omittable_count(self):
        total = sum(ch.no_safely_omittable_count for ch in self.channels)
        return total
    

    def get_underflow_omittable_count(self):
        total = sum(ch.underflow_omittable_count for ch in self.channels)
        return total
    

    def get_overflow_omittable_count(self):
        total = sum(ch.overflow_omittable_count for ch in self.channels)
        return total

    
    def gen_best_check_pos(self, n=2):
        for ch in self.channels:
            ch.gen_best_check_pos(n)

# redundancy omitting with clamping prediction {
    def gen_best_predictive_check_pos(self, clamping_prediction_confidence):
        for ch in self.channels:
            ch.gen_best_predictive_check_pos(clamping_prediction_confidence)
# }


class ModelStats:
    def __init__(self, name):
        self.name = name
        self.operators = []


    def add_operator(self, op_no, op_type):
        if not self.__has_operator(op_no):
            print("ADDING OP", op_no, op_type)
            self.operators.append(LayerStats(op_no, op_type))
        #else: raise RuntimeError("Duplicate OP number")
    

    def add_data_to_op(self, op_no, ch, termination_type, clamping_type, 
                       total, terminated, effectless, safely_omittable):
        op = self.operators[op_no]
        if op == None: raise RuntimeError("Impossible")
        # print("ADDING TO OP", op_no)
        op.add_profiling_data(ch, termination_type, clamping_type, 
                              total, terminated, effectless, safely_omittable)


# redundancy omitting with clamping prediction {
    def add_trace_to_op(self, op_no, ch, clamp_type, output, trace):
        op = self.operators[op_no]
        if op == None: raise RuntimeError("Impossible")
        op.add_trace(ch, clamp_type, output, trace)
# }

    
    def get_total_steps(self):
        total = sum(op.get_total_steps() for op in self.operators)
        return total
    

    def get_terminated_steps(self):
        total = sum(op.get_terminated_steps() for op in self.operators)
        return total
    

    def get_effectless_steps(self):
        total = sum(op.get_effectless_steps() for op in self.operators)
        return total
    

    def get_safely_omittable_steps(self):
        total = sum(op.get_safely_omittable_steps() for op in self.operators)
        return total
    

    def get_neuron_count(self):
        total = sum(op.get_neuron_count() for op in self.operators)
        return total
    

    def get_clamped_neuron_count(self):
        total = sum(op.get_clamped_neuron_count() for op in self.operators)
        return total
    

    def get_underflow_clamping_count(self):
        total = sum(op.get_underflow_clamping_count() for op in self.operators)
        return total
    

    def get_overflow_clamping_count(self):
        total = sum(op.get_overflow_clamping_count() for op in self.operators)
        return total
    

    def get_underflow_termination_count(self):
        total = sum(op.get_underflow_termination_count() for op in self.operators)
        return total
    

    def get_overflow_termination_count(self):
        total = sum(op.get_overflow_termination_count() for op in self.operators)
        return total
    

    def get_no_safely_omittable_count(self):
        total = sum(op.get_no_safely_omittable_count() for op in self.operators)
        return total
    

    def get_underflow_omittable_count(self):
        total = sum(op.get_underflow_omittable_count() for op in self.operators)
        return total


    def get_overflow_omittable_count(self):
        total = sum(op.get_overflow_omittable_count() for op in self.operators)
        return total
    

    def get_dist(self):
        ret = []
        for op in self.operators:
            ret.append(op.get_dist())
        return ret
    

    def gen_best_check_pos(self, n=2):
        for op in self.operators:
            op.gen_best_check_pos(n)

# redundancy omitting with clamping prediction {
    def gen_best_predictive_check_pos(self, clamping_prediction_confidence):
        for op in self.operators:
            op.gen_best_predictive_check_pos(clamping_prediction_confidence)
# }


    def __has_operator(self, op_no: int):
        for op in self.operators:
            if op.idx == op_no: return True
        return False
    

class ProfilingStatsParser:
    def __init__(self):
        self.model = None
        self.parsed_input_file_no = set()
        self.__tmp_cur_working_op = None
        self.parsed_count = 0
        self.correct_count = 0
    

    ################ User interface functions #############
    def display_dist(self):
        dist = self.model.get_dist()
        for i, op in enumerate(dist):
            for j, ch in enumerate(op):
                print(i, j, ch)


    def display_acc(self):
        print("Acc: {:d} / {:d} = {:.2f}".format(
            self.correct_count,
            self.parsed_count,
            self.correct_count / self.parsed_count * 100
        ))


    def display_stats(self):
        sum_total = self.model.get_total_steps()
        sum_term = self.model.get_terminated_steps()
        sum_effectless = self.model.get_effectless_steps()
        sum_s_o = self.model.get_safely_omittable_steps()
        model_total_steps = self.model.get_total_steps()
        sum_neuron_cnt = self.model.get_neuron_count()
        sum_clamped_neuron_cnt = self.model.get_clamped_neuron_count()
        sum_underflow_clamping_cnt = self.model.get_underflow_clamping_count()
        sum_overflow_clamping_cnt = self.model.get_overflow_clamping_count()
        for op in self.model.operators:
            if op.channels == []: continue
            total = op.get_total_steps()
            term = op.get_terminated_steps()
            effectless = op.get_effectless_steps()
            s_o = op.get_safely_omittable_steps()
            neuron_cnt = op.get_neuron_count()
            clamped_neuron_cnt = op.get_clamped_neuron_count()
            no_safely_omittable_cnt = op.get_no_safely_omittable_count()
            # underflow_omittable_cnt = op.get_underflow_omittable_count()
            # overflow_omittable_cnt = op.get_overflow_omittable_count()
            # sum_total += total
            # sum_term += term
            # sum_effectless += effectless
            # sum_s_o += s_o
            print("OP {:d} ({:s}, {:.2f}% of all): \n\tterminated {:d} / {:d} = {:.2f}%, "
                  "safely omittable {:d} / {:d} = {:.2f}%, "
                  "effectless = {:d} / {:d} = {:.2f}%, "
                  "\n\tclamped neuron = {:d} / {:d} = {:.2f}%, "
                  "\n\tnot safely-omittable neuron = {:d} / {:d} = {:.2f}%, "
                #   "underflow omittable neuron = {:d} / {:d} = {:.2f}%, "
                #   "overflow omittable neuron = {:d} / {:d} = {:.2f}%, "
                  "".format(
                op.idx, op.type, total / model_total_steps * 100,
                term, total, term/total*100,
                s_o, total, s_o/total*100,
                effectless, total, effectless/total*100,
                clamped_neuron_cnt, neuron_cnt, clamped_neuron_cnt/neuron_cnt*100,
                no_safely_omittable_cnt, neuron_cnt, no_safely_omittable_cnt/neuron_cnt*100,
                # underflow_omittable_cnt, neuron_cnt, underflow_omittable_cnt/neuron_cnt*100,
                # overflow_omittable_cnt, neuron_cnt, overflow_omittable_cnt/neuron_cnt*100
            ))
        print("Total: terminated {:d} / {:d} = {:.2f}%, "
                "safely omittable {:d} / {:d} = {:.2f}%, "
                "effectless = {:d} / {:d} = {:.2f}%, "
                "clamped neuron = {:d} / {:d} = {:.2f}%, "
                "underflow clamp pct = {:d} / {:d} = {:.2f}%, "
                "overflow clamp pct = {:d} / {:d} = {:.2f}%, "
                "".format(
            sum_term, sum_total, 0 if (sum_total == 0) else (sum_term/sum_total*100),
            sum_s_o, sum_total, 0 if (sum_total == 0) else sum_s_o/sum_total*100,
            sum_effectless, sum_total, 0 if (sum_total == 0) else sum_effectless/sum_total*100,
            sum_clamped_neuron_cnt, sum_neuron_cnt, sum_clamped_neuron_cnt / sum_neuron_cnt * 100,
            sum_underflow_clamping_cnt, sum_clamped_neuron_cnt, 0 if (sum_clamped_neuron_cnt == 0) else (sum_underflow_clamping_cnt/sum_clamped_neuron_cnt*100),
            sum_overflow_clamping_cnt, sum_clamped_neuron_cnt, 0 if (sum_clamped_neuron_cnt == 0) else (sum_overflow_clamping_cnt/sum_clamped_neuron_cnt*100),
        ))
            # print(op.idx, op.type, op.get_total_steps(), op.get_terminated_steps())
    

    def gen_best_check_pos_file(self, n, fname):
        self.model.gen_best_check_pos(n)
        per_layer = []
        for op in self.model.operators:
            per_channel = []
            for ch in op.channels:
                per_channel.append(ch.best_check_pos)
            per_layer.append(per_channel)
        with open(fname, 'wb') as f:
            pickle.dump(per_layer, f)


# redundancy omitting with clamping prediction {
    def gen_best_predictive_check_pos_file(self, 
                                           clamping_prediction_confidence, 
                                           fname):
        self.model.gen_best_predictive_check_pos(
                clamping_prediction_confidence)
        per_layer_pos = []
        per_layer_threshold = []
        per_layer_type = []
        for op_id, op in enumerate(self.model.operators):
            per_channel_pos = []
            per_channel_threshold = []
            per_channel_type = []
            for ch_idx, ch in enumerate(op.channels):
                per_channel_pos.append(ch.best_predictive_check_pos)
                per_channel_threshold.append(
                    [ch.per_step_predictive_termination_threshold[i] \
                            for i in ch.best_predictive_check_pos]
                )
                per_channel_type.append(
                    [ch.per_step_predictive_termination_type[i] \
                            for i in ch.best_predictive_check_pos]
                )
            per_layer_pos.append(per_channel_pos)
            per_layer_threshold.append(per_channel_threshold)
            per_layer_type.append(per_channel_type)
        print(per_layer_pos)
        print(per_layer_type)
        print(per_layer_threshold)
        with open(fname, 'wb') as f:
            pickle.dump({
                "per_layer_pos": per_layer_pos,
                "per_layer_threshold": per_layer_threshold,
                "per_layer_type": per_layer_type,
            }, f)
# }


    ################ Parsing functions ###############
    def parse_profile(self, fname: str):
        with open(fname) as f:
            self.parse_first_line(f.readline())
            for line in f:
                line = line.strip()
                if len(line) == 0: continue
                if line.startswith('='):
                    self.parse_op_line(line)
                elif line.startswith('ch'):
                    self.parse_line(line)
# redundancy omitting with clamping prediction {
                elif line.startswith('tr'):
                    self.parse_comp_trace(line)
# }
                elif line.startswith('Label'):
                    self.parse_acc(line)
        self.__tmp_cur_working_op = None
    

    def parse_first_line(self, s: str):
        m = re.search(r'^Starting ([^,]+), input (\d+)$', s.strip())
        name = m.group(1)
        no = int(m.group(2))
        if self.model == None:
            self.model = ModelStats(name)
        elif self.model.name != name: raise RuntimeError("Another model!?")
        if no not in self.parsed_input_file_no:
            self.parsed_input_file_no.add(no)
        else: raise RuntimeError("Already parsed")
    

    def parse_op_line(self, s: str):
        m = re.search(r'^=== OP (\d+), ([^=]*) ===', s.strip())
        op_no = int(m.group(1))
        op_type = m.group(2)
        self.model.add_operator(op_no, op_type)
        self.__tmp_cur_working_op = op_no #len(self.model.operators) - 1
        # print("OP_NO and TYPE:", op_no, op_type)

    
    def parse_line(self, s: str):
        termination_type_translate = {0: "None", 1: "Underflow", 2: "Overflow"}

        # m = re.search(r'^channel (\d+)\|termination_type (\d+)' \
        #               + r'\|termination_step (\d+)\|total_step (\d+)' \
        #               + r'\|clamping_type (\d+)\|clamping_start (\d+)' \
        #               + r'\|false_clamping (\d+)\|is_safely_omittable (\d+)' \
        #               + r'\|safely_omittable_start (\d+)\|non_clamped_out (-?\d+)' \
        #               + r'\|clamped_out (-?\d+)', s.strip())
        m = re.search(r'^ch (\d+)\|(\d+)' \
                      + r'\|(\d+)\|(\d+)' \
                      + r'\|(\d+)\|(\d+)' \
                      + r'\|(\d+)\|(\d+)' \
                      + r'\|(\d+)\|(-?\d+)' \
                      + r'\|(-?\d+)', s.strip())
        channel = int(m.group(1))
        termination_type = termination_type_translate[int(m.group(2))]
        clamping_type = termination_type_translate[int(m.group(5))]
        total = int(m.group(4))
        terminated = total - int(m.group(3))
        effectless = total - int(m.group(6))
        safely_omittable = total - int(m.group(9))
        self.__add_data_to_cur_working_op(channel, termination_type, 
                                          clamping_type,
                                          total, terminated,
                                          effectless, safely_omittable)
    

# redundancy omitting with clamping prediction {
    def parse_comp_trace(self, s: str):
        s = s.split("|")
        channel = int(s[0].split(" ")[1])
        clamp_type = int(s[1])
        output = int(s[2])
        trace = [int(each) for each in s[3:]]
        self.__add_trace_to_cur_working_op(channel, clamp_type, output, trace)
        pass
# }
    

    def parse_acc(self, s: str):
        m = re.search("^Label: (\d+), Max: (\d+)", s.strip())
        self.parsed_count += 1
        if int(m.group(1)) == int(m.group(2)):
            self.correct_count += 1


    ################# Util functions ###############
    def __add_data_to_cur_working_op(self, ch, termination_type, 
                                     clamping_type, total,
                                     terminated, effectless, safely_omittable):
        self.model.add_data_to_op(self.__tmp_cur_working_op, ch,
                                  termination_type, clamping_type,
                                  total, terminated,
                                  effectless, safely_omittable)
        
# redundancy omitting with clamping prediction {
    def __add_trace_to_cur_working_op(self, ch, clamp_type, output, trace):
        self.model.add_trace_to_op(self.__tmp_cur_working_op, ch, 
                                   clamp_type, output, trace)
# }


# Example usage
if __name__ == "__main__":
    fr = ProfilingStatsParser()
    base_path = sys.argv[1]
    mode = sys.argv[2]
    if mode == "display":
        for i in range(352, 384):#320):
        # for i in range(0, 32):
            # print("Parsing profile data for input {:d}".format(i))
            file_path = base_path + "/out_{:d}".format(i)
            fr.parse_profile(file_path)

        fr.display_acc()
        fr.display_stats()
        # fr.display_dist()
    elif mode == "gen_pos":
        samples = ""
        try: 
            with open("profile_data{:s}_no.pkl".format(samples), 'rb') as f:
                print("Opening cached profile data...")
                fr = pickle.load(f)
        except Exception as e:
            print(e)
            print("Cached profile data not found, generating...")
            start_time = time.time()
            for i in range(256, 288):#320, 352):#320 + 4):#352):#383):#352):
            # for i in range(0, 32):
                # print("Parsing profile data for input {:d}".format(i))
                file_path = base_path + "/out_{:d}".format(i)
                fr.parse_profile(file_path)
            spent_time = time.time() - start_time
            print("Spent {:f} seconds on reading profiling data".format(spent_time))
            #TEST
            # with open("profile_data{:s}.pkl".format(samples), 'wb') as f:
            #     pickle.dump(fr, f)

        fr.display_acc()
        fr.display_stats()
        #TEST
        # fr.gen_best_check_pos_file(2, fr.model.name + ".pkl")
    elif mode == "gen_pred":
        clamping_pred_conf_int = int(sys.argv[3])
        clamping_pred_conf = clamping_pred_conf_int / 1000
        print("Running predictive mode with conf {:d} ({:f})".format(clamping_pred_conf_int, clamping_pred_conf))
        samples = "_32"

        # Try and use cached profiled data first
        try: 
            with open("profile_data{:s}.pkl".format(samples), 'rb') as f:
                print("Opening cached profile data...")
                fr = pickle.load(f)
        except Exception as e:
            print(e)
            print("Cached profile data not found, generating...")
            start_time = time.time()
            for i in range(320, 352):#320 + 4):#352):#383):#352):
            # for i in range(0, 32):
                # print("Parsing profile data for input {:d}".format(i))
                file_path = base_path + "/out_{:d}".format(i)
                fr.parse_profile(file_path)
            spent_time = time.time() - start_time
            print("Spent {:f} seconds on reading profiling data".format(spent_time))
            with open("profile_data{:s}.pkl".format(samples), 'wb') as f:
                pickle.dump(fr, f)

        fr.display_acc()
        fr.display_stats()

        start_time = time.time()
        fr.gen_best_predictive_check_pos_file(clamping_pred_conf, fr.model.name 
                            + "pred_{:d}{:s}.pkl".format(clamping_pred_conf_int, samples))
        spent_time = time.time() - start_time
        print("Spent {:f} seconds on parsing profiling data".format(spent_time))
        
    elif mode == "test":
        #with open("profile_data.pkl", 'rb') as f:
        #    fr = pickle.load(f)
        try: 
            with open("profile_data_plot.pkl", 'rb') as f:
                print("Opening cached profile data...")
                fr = pickle.load(f)
        except Exception as e:
            print(e)
            print("Cached profile data not found, generating...")
            for i in range(320, 352):#383):#352):
            # for i in range(0, 32):
                # print("Parsing profile data for input {:d}".format(i))
                file_path = "profile_sat_pred_plot_outputs" + "/out_{:d}".format(i)
                fr.parse_profile(file_path)
            with open("profile_data_plot.pkl", 'wb') as f:
                pickle.dump(fr, f)
        print("-----------------------")
        for i in range(17):
            #print("Channel 6 step {:d} val freq:".format(i))
            print(dict(sorted(fr.model.operators[5].channels[6].steps[i].val_freq.items())), ",")
        print("-----------------------")
        for i in range(17):
            #print("Channel 6 step {:d} val under clamp count:".format(i))
            print(dict(sorted(fr.model.operators[5].channels[6].steps[i].val_under_clamp_count.items())), ",")
        for i in range(17):
            print(fr.model.operators[5].channels[6].steps[i].get_val_to_under_clamp_with_confidence(1.00))
        
        print(fr.model.operators[5].channels[6].steps[8].get_val_to_under_clamp_with_confidence(1.25))


    # s = 2
    # print(fr.model.operators[1].channels[0].steps[s].val_freq)
    # print(fr.model.operators[1].channels[0].steps[s].val_under_clamp_count)
    # print(fr.model.operators[1].channels[0].steps[s].val_over_clamp_count)
    # print(fr.model.operators[1].channels[0].steps[s].get_val_to_under_clamp_with_prob(0.95))
    # print(fr.model.operators[1].channels[0].steps[s].get_val_to_over_clamp_with_prob(0.95))
