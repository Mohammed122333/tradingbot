import re
import json

def fix_ver():
    with open('c:/Users/adeL/Desktop/GoodBot/Ver2/config.py', 'r', encoding='utf-8') as f:
        data = f.read()
    data = re.sub(r'BOT_VERSION\s*=\s*"\d+\.\d+\.\d+"', 'BOT_VERSION = "20.0.1"', data)
    with open('c:/Users/adeL/Desktop/GoodBot/Ver2/config.py', 'w', encoding='utf-8') as f:
        f.write(data)

    with open('c:/Users/adeL/Desktop/GoodBot/Ver2/version_cache.json', 'r', encoding='utf-8') as f:
        cache = json.load(f)
    cache['version'] = '20.0.1'
    with open('c:/Users/adeL/Desktop/GoodBot/Ver2/version_cache.json', 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=4)

if __name__ == '__main__':
    fix_ver()
