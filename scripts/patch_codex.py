import os
import re

cli_js_path = r"C:\Users\temsan\AppData\Roaming\npm\node_modules\@openai\codex\dist\cli.js"

if not os.path.isfile(cli_js_path):
    print(f"Файл не найден: {cli_js_path}")
    exit(1)

with open(cli_js_path, "r", encoding="utf-8") as f:
    content = f.read()

# Патчим имя лог-файла — заменяем двоеточия в toISOString()
patched_content = re.sub(
    r"(new Date\(\)\.toISOString\(\))",
    r'new Date().toISOString().replace(/:/g, "-")',
    content
)

if content == patched_content:
    print("Не найден код для патча.")
    exit(1)

backup_path = cli_js_path + ".backup"
os.rename(cli_js_path, backup_path)
print(f"Создан бэкап: {backup_path}")

with open(cli_js_path, "w", encoding="utf-8") as f:
    f.write(patched_content)

print("Патч применён успешно.")
