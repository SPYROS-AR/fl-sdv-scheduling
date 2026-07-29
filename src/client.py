import flwr
from flwr.client import ClientApp
from flwr.common import Context
import model


from src import constants

class FLVehicle(flwr.client.NumPyClient):

    def __init__(self, partition_id: int, model_size_mb: float = 50.0, 
                 epochs_per_compute: int = 10):
        # System model
        self.id = partition_id
        self.M = model_size_mb # M_i
        self.rho = epochs_per_compute # \rho_i

        self.prev_loss = 0.0 # L_i(t-1)
    
        self.current_drift = 0.0 # D_i(t)
        
        # CNN model
        self.model = model.flCNN()
    
    # Fl methods
    def get_parameters(self, config):
        # Return model parameters as a list of NumPy arrays
        return []

    def fit(self, parameters, config):
        # Simulate local training and return updated model parameters
        return [], len(parameters), {}

    def evaluate(self, parameters, config):
        # Simulate evaluation and return loss and metrics
        return 0.0, len(parameters), {}
    
    # System model methods
    def _predict_loss(self, loss_history):
        pass
            
    
    def _compute_drift(self, global_weights):
        pass
    
    def _compute_utility(self, gain, drift):
        pass
    
    def _set_parameters(self, paramaters):
        pass
    

def client_fn(context: Context):

    vehicle_id = int(context.node_config["partition-id"])

    return FLVehicle(
        partition_id=vehicle_id,
        model_size_mb=50.0,
        epochs_per_compute=5
    ).to_client()
    
app = ClientApp(client_fn=client_fn)