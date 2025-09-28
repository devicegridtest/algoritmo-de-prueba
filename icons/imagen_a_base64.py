import base64

with open("icons/logo1.png", "rb") as imagen:
    cadena_base64 = base64.b64encode(imagen.read()).decode('utf-8')

print(cadena_base64)