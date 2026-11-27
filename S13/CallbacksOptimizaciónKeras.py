import tensorflow as tf
from keras import callbacks, optimizers

# Optimizador AdamW con decaimiento de pesos (regularización L2 desacoplada)
optimizador = optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)

# Callbacks para controlar el entrenamiento automáticamente
lista_callbacks = [
    # Detiene el entrenamiento si val_loss no mejora en 5 épocas y restaura los mejores pesos
    callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    
    # Reduce el learning rate a la mitad si val_loss se estanca por 2 épocas
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
    
    # Guarda automáticamente el mejor modelo en disco
    callbacks.ModelCheckpoint(filepath='mejor_modelo.keras', monitor='val_accuracy', save_best_only=True)
]

# Ejemplo de uso en entrenamiento:
# modelo.compile(optimizer=optimizador, loss='categorical_crossentropy', metrics=['accuracy'])
# modelo.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50, callbacks=lista_callbacks)