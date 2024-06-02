class ValorInvalidoException(Exception):
    """Excepción lanzada cuando se encuentra un valor inválido."""
    
    def __init__(self, mensaje="Error: Tipo de valor inválido"):
        super().__init__(mensaje)

class LongitudInvalidaException(Exception):
    """Excepción lanzada cuando las longitudes no coinciden."""
    
    def __init__(self, mensaje="Error: Longitud de y_true y y_pred no coinciden"):
        super().__init__(mensaje)

class AtributoNoEncontradoException(Exception):
    """Excepción lanzada cuando un atributo no se encuentra en los datos."""
    
    def __init__(self, mensaje="Error: Atributo no encontrado en los datos"):
        super().__init__(mensaje)

class ConversionDeTiposException(Exception):
    """Excepción lanzada cuando se produce una conversión de tipos inválida."""
    
    def __init__(self, mensaje="Error: Conversión de tipos inválida"):
        super().__init__(mensaje)

class GraficadorException(Exception):
    """Excepción base para errores en la clase Graficador."""
    def __init__(self, mensaje="Error en el graficador"):
        self.mensaje = mensaje
        super().__init__(self.mensaje)




