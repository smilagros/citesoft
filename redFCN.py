import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Ejemplo de uso:
input_size = 10
hidden_size = 20
output_size = 1

# Crear instancia de la red neuronal
model = FCN(input_size, hidden_size, output_size)

# Generar un tensor de ejemplo
input_data = torch.randn(1, input_size)

# Pasar el tensor a través del modelo
output = model(input_data)

print("Output:", output)
