import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


class MultiModalNet(nn.Module):
    """
    Red neuronal multimodal para procesamiento de imágenes médicas y textos de historias clínicas.

    Args:
        image_input_size (int): Tamaño de las imágenes de entrada.
        text_input_size (int): Tamaño del vector de palabras de entrada.
        hidden_size (int): Tamaño del espacio latente.
        output_size (int): Número de clases de salida.
        embedding_dim (int): Dimensión de los embeddings de palabras.
    """

    def __init__(self, image_input_size, text_input_size, hidden_size, output_size, embedding_dim):
        super(MultiModalNet, self).__init__()

        # Capas para procesamiento de imágenes
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc_image = nn.Linear(64 * 64 * 64, hidden_size)  # Corregido

        # Capas para procesamiento de texto
        self.embedding = nn.Embedding(text_input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=2, batch_first=True)

        # Capas de fusión
        self.fc_fusion = nn.Linear(hidden_size * 2, hidden_size)

        # Capa de salida
        self.fc_output = nn.Linear(hidden_size, output_size)

    def forward(self, images, texts):
        """
        Propagación hacia adelante de los datos a través de la red multimodal.

        Args:
            images (torch.Tensor): Batch de imágenes médicas de entrada.
            texts (torch.Tensor): Batch de secuencias de texto de historias clínicas de entrada.

        Returns:
            torch.Tensor: Salida de la red neuronal.
        """
        # Procesamiento de imágenes
        x_image = F.relu(self.conv1(images))
        x_image = F.relu(self.conv2(x_image))
        x_image = torch.flatten(x_image, start_dim=1)
        x_image = F.relu(self.fc_image(x_image))

        # Procesamiento de texto
        x_text = self.embedding(texts)
        _, (x_text, _) = self.lstm(x_text)
        x_text = x_text[-1]  # Tomar solo la última salida de la secuencia

        # Fusionar características
        x_fusion = torch.cat((x_image, x_text), dim=1)
        x_fusion = F.relu(self.fc_fusion(x_fusion))

        # Capa de salida
        output = self.fc_output(x_fusion)
        return output


# Definir los tamaños de entrada y salida
image_input_size = 64 * 64  # Tamaño de las imágenes de entrada (64x64 píxeles)
text_input_size = 100  # Tamaño del vector de palabras de entrada
hidden_size = 128  # Tamaño del espacio latente
output_size = 2  # Número de clases de salida
embedding_dim = 50  # Dimensión de los embeddings de palabras

# Crear una instancia del modelo multimodal
model = MultiModalNet(image_input_size, text_input_size, hidden_size, output_size, embedding_dim)

# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Datos de ejemplo (simulados)
images = torch.randn(32, 3, 64, 64)  # Batch de 32 imágenes (canales, altura, anchura)
texts = torch.randint(0, 100, (32, 20))  # Batch de 32 secuencias de texto (longitud máxima 20)

# Etiquetas de ejemplo (simuladas)
labels = torch.randint(0, 2, (32,))  # Batch de 32 etiquetas (0 o 1)

# Crear conjuntos de datos y cargadores de datos
dataset = TensorDataset(images, texts, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Entrenamiento del modelo
num_epochs = 5

for epoch in range(num_epochs):
    for batch_images, batch_texts, batch_labels in dataloader:
        # Reiniciar los gradientes
        optimizer.zero_grad()

        # Propagar hacia adelante
        outputs = model(batch_images, batch_texts)

        # Calcular la pérdida
        loss = criterion(outputs, batch_labels)

        # Propagar hacia atrás y actualizar los pesos
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

# Uso del modelo para predicción
# Supongamos que tienes nuevos datos de imágenes y textos (new_images, new_texts)
new_images = torch.randn(5, 3, 64, 64)  # Por ejemplo, 5 nuevas imágenes
new_texts = torch.randint(0, 100, (5, 20))  # Por ejemplo, 5 nuevas secuencias de texto

# Pasar los nuevos datos a través del modelo
with torch.no_grad():
    predictions = model(new_images, new_texts)

for epoch in range(num_epochs):
    for batch_images, batch_texts, batch_labels in dataloader:
        # Reiniciar los gradientes
        optimizer.zero_grad()

        # Propagar hacia adelante
        outputs = model(batch_images, batch_texts)

        # Calcular la pérdida
        loss = criterion(outputs, batch_labels)

        # Propagar hacia atrás y actualizar los pesos
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

# Uso del modelo para predicción
# Supongamos que tienes nuevos datos de imágenes y textos (new_images, new_texts)
new_images = torch.randn(5, 3, 64, 64)  # Por ejemplo, 5 nuevas imágenes
new_texts = torch.randint(0, 100, (5, 20))  # Por ejemplo, 5 nuevas secuencias de texto

# Pasar los nuevos datos a través del modelo
with torch.no_grad():
    predictions = model(new_images, new_texts)
    _, predicted_labels = torch.max(predictions, 1)

print("Predicted labels for new data:", predicted_labels)
