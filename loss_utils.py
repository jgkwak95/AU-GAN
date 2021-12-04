import tensorflow as tf

epsilon = 1e-7

def conf_criterion_lp(im1, im2, conf_sigma):  # factorized laplacian distribution
    loss = tf.abs(im1 - im2)
    if conf_sigma is not None:
        loss = loss * 2 / (conf_sigma + epsilon) + tf.log(conf_sigma * 2 + epsilon)
        loss = tf.reduce_mean(loss)
    else:
        loss = tf.reduce_mean(loss)

    return loss



def conf_criterion(im1, im2, conf_sigma):  # gaussian distribution
    loss = tf.abs(im1 - im2)
    if conf_sigma is not None:
        loss = tf.math.exp(-conf_sigma) * 5 * loss + conf_sigma / 2
        loss = tf.reduce_mean(loss)
    else:
        loss = tf.reduce_mean(loss)

    return loss


def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_ - target) ** 2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


def mae_criterion_list(in_, target):
    loss = 0.0
    for i in range(len(target)):
        loss += tf.reduce_mean((in_[i] - target[i]) ** 2)
    return loss / len(target)


def sce_criterion_list(logits, labels):
    loss = 0.0
    for i in range(len(labels)):
        loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits[i], labels=labels[i]))
    return loss / len(labels)