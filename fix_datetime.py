import os
import re

FILES_TO_FIX = ["app_realtime.py", "predict_model.py", "backtest_engine.py", "train_model.py"]

def fix_datetime_imports(filepath):
    if not os.path.exists(filepath):
        print(f"⚠️ No encontrado: {filepath}")
        return
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    original = content
    
    # Eliminar 'import datetime' suelto
    content = re.sub(r'^import datetime\s*$', '', content, flags=re.MULTILINE)
    
    # Asegurar import correcto
    if 'from datetime import datetime, timedelta' not in content:
        content = content.replace('import time\n', 'import time\nfrom datetime import datetime, timedelta\n', 1)
    
    # Reemplazos críticos
    content = content.replace('datetime.datetime.now()', 'datetime.now()')
    content = content.replace('datetime.timedelta', 'timedelta')
    content = content.replace('datetime.timezone', 'timezone')
    
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Corregido: {filepath}")
    else:
        print(f"ℹ️ OK: {filepath}")

print("🔧 Corrigiendo datetime...\n")
for f in FILES_TO_FIX:
    fix_datetime_imports(f)
print("\n✨ Listo!")
