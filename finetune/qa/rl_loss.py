import tensorflow as tf


def compute_loss(logits, positions, seq_length):
    one_hot_positions = tf.one_hot(
        positions, depth=seq_length, dtype=tf.float32)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    loss = -tf.reduce_sum(one_hot_positions * log_probs, axis=-1)
    return loss


def simple_tf_f1_score(tensors):
    prediction_start = tf.cast(tensors[0], dtype=tf.float32)
    prediction_end = tf.cast(tensors[1], dtype=tf.float32)
    ground_truth_start = tf.cast(tensors[2], dtype=tf.float32)
    ground_truth_end = tf.cast(tensors[3], dtype=tf.float32)

    min_end = tf.reduce_min([prediction_end, ground_truth_end])
    max_start = tf.reduce_max([prediction_start, ground_truth_start])

    overlap = tf.cond(tf.greater(max_start, min_end), lambda: 0., lambda: min_end - max_start + 1)
    precision = tf.cond(tf.equal(overlap, 0.), lambda: 0., lambda: overlap / (prediction_end - prediction_start + 1))
    recall = tf.cond(tf.equal(overlap, 0.), lambda: 1e-20,
                     lambda: overlap / (ground_truth_end - ground_truth_start + 1))

    f1 = (2 * precision * recall) / (precision + recall)

    # f1 = tf.cond(tf.greater(prediction_start, prediction_end), lambda: 0., lambda: f1)
    # f1 = tf.cond(tf.equal(ground_truth_end, 0) & ~tf.equal(prediction_end, 0), lambda: 0., lambda: f1)
    return f1


def reward(guess_start, guess_end, answer_start, answer_end, baseline, sample_num):
    """
    Reinforcement learning reward (i.e. F1 score) from sampling a trajectory of guesses across each decoder timestep
    """
    reward = [[]] * sample_num

    for t in range(sample_num):
        f1_score = tf.map_fn(
            simple_tf_f1_score, (guess_start[:, t], guess_end[:, t], answer_start, answer_end),
            dtype=tf.float32)  # [bs,]
        normalized_reward = tf.stop_gradient(f1_score - baseline)
        reward[t] = normalized_reward
    return tf.stack(reward, axis=-1)  # [bs, sample]


def surrogate_loss(start_logits, end_logits, guess_start, guess_end, r, seq_length, num_samples):
    """
    The surrogate loss to be used for policy gradient updates
    """
    bsz = start_logits.shape.as_list()[0]

    guess_start = tf.reshape(guess_start, [-1])  # (bs * simple_num ,)
    guess_end = tf.reshape(guess_end, [-1])
    r = tf.reshape(r, [-1])
    start_logits = tf.concat(
        [tf.tile(_sp, [num_samples, 1]) for _sp in tf.split(start_logits, bsz)], axis=0)
    end_logits = tf.concat(
        [tf.tile(_sp, [num_samples, 1]) for _sp in tf.split(end_logits, bsz)], axis=0)
    start_loss = r * compute_loss(start_logits, guess_start, seq_length)
    end_loss = r * compute_loss(end_logits, guess_end, seq_length)
    start_loss = tf.reduce_mean(tf.stack(tf.split(start_loss, num_samples), axis=1), axis=1)
    end_loss = tf.reduce_mean(tf.stack(tf.split(end_loss, num_samples), axis=1), axis=1)
    return start_loss, end_loss


# def rl_loss(start_logits, end_logits, answer_start, answer_end, sample_num=4):
#     """
#     Reinforcement learning loss
#     """
#     guess_start_greedy = tf.argmax(start_logits, axis=1)
#     guess_end_greedy = tf.argmax(end_logits, axis=1)
#     baseline = tf.map_fn(simple_tf_f1_score, (guess_start_greedy, guess_end_greedy,
#                                               answer_start, answer_end), dtype=tf.float32)
#     baseline = tf.math.minimum(baseline, 0.8)
#
#     guess_start = []
#     guess_end = []
#
#     guess_start.append(tf.multinomial(start_logits, sample_num))
#     guess_end.append(tf.multinomial(end_logits, sample_num))
#     guess_start = tf.concat(guess_start, axis=0)
#     guess_end = tf.concat(guess_end, axis=0)
#     r = reward(guess_start, guess_end, answer_start, answer_end, baseline, sample_num)  # [bs*project_layers,4]
#     # print("reward_shape:", r.shape)
#     surr_loss = surrogate_loss(start_logits, end_logits, guess_start, guess_end, r, sample_num)
#     loss = tf.reduce_mean(-r)
#
#     # This function needs to return the value of loss in the forward pass so that theta_rl gets the right parameter update
#     # However, this needs to have the gradient of surr_loss in the backward pass so the model gets the right policy gradient update
#     return surr_loss + tf.stop_gradient(loss - surr_loss)


def sample_with_greedy(start_logits, end_logits, guess_start_greedy, guess_end_greedy, seq_length, num_samples):
    start_seq_mask = tf.sequence_mask(guess_start_greedy, maxlen=seq_length, dtype=tf.float32)
    end_seq_mask = tf.sequence_mask(guess_end_greedy, maxlen=seq_length, dtype=tf.float32)

    mask_start_logits = start_logits + start_seq_mask * -1e30 + (1 - end_seq_mask) * -1e30
    mask_end_logits = end_logits + start_seq_mask * -1e30 + (1 - end_seq_mask) * -1e30

    guess_starts = tf.random.categorical(mask_start_logits, num_samples, dtype=tf.int32)
    guess_ends = tf.random.categorical(mask_end_logits, num_samples, dtype=tf.int32)
    return guess_starts, guess_ends


def reforce_f1_ce_loss(start_logits, end_logits, start_positions, end_positions, num_samples):
    guess_start_greedy = tf.argmax(start_logits, axis=1, output_type=tf.int32)
    guess_end_greedy = tf.argmax(end_logits, axis=1, output_type=tf.int32)

    baseline = tf.map_fn(simple_tf_f1_score, (guess_start_greedy, guess_end_greedy,
                                              start_positions, end_positions), dtype=tf.float32)  # [bs, ]

    bsz, seq_length = start_logits.shape.as_list()

    is_em = tf.logical_and(tf.equal(start_positions, guess_start_greedy), tf.equal(end_positions, guess_end_greedy))
    is_no_answer = tf.logical_and(tf.equal(start_positions, 0), tf.equal(end_positions, 0))  # [bs, ]
    is_contain_answer = tf.logical_and(tf.greater_equal(guess_start_greedy, start_positions),
                                       tf.less_equal(guess_end_greedy, end_positions))

    start_ce_loss = compute_loss(start_logits, start_positions, seq_length)
    end_ce_loss = compute_loss(end_logits, end_positions, seq_length)

    guess_starts, guess_ends = sample_with_greedy(start_logits, end_logits, guess_start_greedy, guess_end_greedy,
                                                  seq_length=seq_length, num_samples=num_samples)  # [bs, num_samples]
    r = reward(guess_starts, guess_ends, start_positions, end_positions, baseline, num_samples)  # [bs, num_samples]
    start_rl_loss, end_rl_loss = surrogate_loss(start_logits, end_logits, guess_starts, guess_ends, r, seq_length,
                                                num_samples)

    gamma = tf.cond(tf.logical_and(tf.logical_not(tf.logical_or(is_em, is_no_answer)), is_contain_answer),
                    lambda: .5 * tf.ones_like(start_ce_loss),
                    lambda: tf.zeros_like(start_ce_loss))
    start_loss = start_ce_loss * (1 - gamma) + start_rl_loss * gamma
    end_loss = end_ce_loss * (1 - gamma) + end_rl_loss * gamma

    return (start_loss + end_loss) / 2.
