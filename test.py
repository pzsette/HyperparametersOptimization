import rbfopt
import matplotlib.pyplot as plt
from model import *
from model2 import *
from keras import datasets


# bonmin and ipot located in System path usr/local/bin
settings = rbfopt.RbfoptSettings(minlp_solver_path='bonmin', nlp_solver_path='ipopt')

# Max evaluations number of the objective function
num_evaluations = 10

# Bounds for each each hyperparameter
hyp_bounds = {"learning_rate": (0.0001, 0.1), "decay": (0, 0.1)}


def evaluate_hyp(hyperparameters):

    learning_rate = hyperparameters[0]
    decay = hyperparameters[1]

    nn_model = Model(learning_rate, decay, train_images, train_labels, test_images, test_labels)
    score, accuracy = nn_model.evaluate_model()

    lr_history.append(learning_rate)
    d_history.append(decay)
    test_loss_history.append(score)
    test_accuracy_history.append(accuracy)

    return score


def evaluate_hyp2(hyperparameters):

    learning_rate = hyperparameters[0]
    decay = hyperparameters[1]

    nn_model = Model2(learning_rate, decay)
    results = nn_model.evaluate_model()

    lr_history.append(learning_rate)
    d_history.append(decay)
    test_loss_history.append(results[0])
    test_accuracy_history.append(results[1])

    return results[0]


if __name__ == '__main__':

    # Read Dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

    # Get lower and upper bound for each variable
    lower_bounds = [hyp_bounds["learning_rate"][0], hyp_bounds["decay"][0]]
    upper_bounds = [hyp_bounds["learning_rate"][1], hyp_bounds["decay"][1]]

    # MODEL TEST

    lr_history = []
    d_history = []
    m_history = []
    test_loss_history = []
    test_accuracy_history = []

    # Scale to a range from 0 to 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Build RBF optimizer
    bb = rbfopt.RbfoptUserBlackBox(2, np.array(lower_bounds), np.array(upper_bounds), np.array(['R', 'R']), evaluate_hyp)
    settings = rbfopt.RbfoptSettings(num_evaluations, target_objval=0.0)
    alg = rbfopt.RbfoptAlgorithm(settings, bb)
    val, x, itercount, evalcount, fast_evalcount = alg.optimize()

    print("\nRESUME TRAINING FASHION-MNIST DATASET:")
    print("      Learningr Rate        Decay            Test loss       Test accuracy")
    for i in range(num_evaluations + 3):
        print([str(lr_history[i]) + ', ' + str(d_history[i]) + ', ' + str(test_loss_history[i]) + ', ' + str(test_accuracy_history[i])])
    print("\nResults with RBF optimizer: " + str({"Test loss": val, "learning_rate": x[0], "weight_decay": x[1]}) + "\n")

    plt.plot(np.arange(num_evaluations+3), test_loss_history, marker='o', color='b', label='dataset fashion MNIST')

    # MODEl2 TEST

    lr_history = []
    d_history = []
    m_history = []
    test_loss_history = []
    test_accuracy_history = []

    # Build RBF optimizer
    bb = rbfopt.RbfoptUserBlackBox(2, np.array(lower_bounds), np.array(upper_bounds), np.array(['R', 'R']), evaluate_hyp2)
    settings = rbfopt.RbfoptSettings(num_evaluations, target_objval=0.0)
    alg = rbfopt.RbfoptAlgorithm(settings, bb)
    val, x, itercount, evalcount, fast_evalcount = alg.optimize()

    print("\nRESUME TRAINING PIMA DATASET:")
    print("      Learning Rate        Decay            Test loss            Test accuracy")
    for i in range(num_evaluations + 3):
        print([str(lr_history[i]) + ', ' + str(d_history[i]) + ', ' + str(test_loss_history[i]) + ', ' + str(test_accuracy_history[i])])
    print("\nResults with RBF optimizer: " + str({"Test loss": val, "learning_rate": x[0], "weight_decay": x[1]}))

    # Create plot
    plt.plot(np.arange(num_evaluations+3), test_loss_history, marker='o', color='g', label='dataset PIMA')
    plt.legend(loc="lower right")
    plt.xlabel('Iteration')
    plt.xticks(np.arange(num_evaluations+3))
    plt.ylabel('Test loss value')

    plt.ylim(top=0.8)
    plt.ylim(bottom=0.2)

    plt.axvline(x=2.5, color='r', linestyle='--')

    plt.show()
