
import time
import pickle
import yaml
import threading


from torch.utils.tensorboard import SummaryWriter

from src.server import *
from src.utils import launch_tensor_board


if __name__ == "__main__":
    # read configuration file
    with open('config.yaml') as c:
        configs = list(yaml.load_all(c, Loader=yaml.FullLoader))
    global_config = configs[0]["global_config"]
    data_config = configs[1]["data_config"]
    fed_config = configs[2]["fed_config"]
    optim_config = configs[3]["optim_config"]
    init_config = configs[4]["init_config"]
    model_config = configs[5]["model_config"]
    log_config = configs[6]["log_config"]
   
    # modify log_path to contain current time
    log_config["log_path"] = os.path.join(log_config["log_path"], str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

    # initiate TensorBaord for tracking losses and metrics
    writer = SummaryWriter(log_dir="log", filename_suffix="FL")
    tb_thread = threading.Thread(
        target=launch_tensor_board,
        args=([log_config["log_path"], log_config["tb_port"], log_config["tb_host"]])
        ).start()
    time.sleep(3.0)

    # set the configuration of global logger
    logger = logging.getLogger(__name__)

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%a %d %b %Y %H:%M:%S', filename='my.log', filemode='w')

    
    # display and log experiment configuration
    message = "\n[WELCOME] Unfolding configurations...!"
    print(message); logging.info(message)

    for config in configs:
        print(config); logging.info(config)
    print()

    # initialize federated learning 
    central_server = Server(writer, model_config, global_config, data_config, init_config, fed_config, optim_config)
    central_server.setup()

    # do federate learning
    central_server.fit()

    # save resulting losses and metrics
    with open(os.path.join(log_config["log _path"], "result.pkl"), "wb") as f:
        pickle.dump(central_server.results, f)
    
    # bye!
    message = "...done all learning process!\n...exit program!"
    print(message); logging.info(message)
    time.sleep(3); exit()

