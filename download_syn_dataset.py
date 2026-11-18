import os
import sys
import ssl
import time
import urllib.request

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

DATASET_DIR = r"C:\Study\HocKy6\MH&MM\qml-ddos\dataset\CICDDoS2019\CSVs"

FILES = [
    {
        "name": "01-12/Syn.csv (Train)",
        "url": "https://huggingface.co/datasets/baalajimaestro/CICDDoS2019/resolve/main/01-12/Syn.csv",
        "dest": os.path.join(DATASET_DIR, "01-12", "Syn.csv"),
    },
    {
        "name": "03-11/Syn.csv (Test)",
        "url": "https://huggingface.co/datasets/baalajimaestro/CICDDoS2019/resolve/main/03-11/Syn.csv",
        "dest": os.path.join(DATASET_DIR, "03-11", "Syn.csv"),
    }
]

def get_remote_file_size(url, ctx):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}, method='HEAD')
    with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
        cl = resp.headers.get('Content-Length')
        return int(cl) if cl else None

def download_file_with_resume(item):
    name = item["name"]
    url = item["url"]
    dest = item["dest"]

    os.makedirs(os.path.dirname(dest), exist_ok=True)

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    print(f"[CHECK] Dang kiem tra kich thuoc {name} tu server...")
    remote_size = get_remote_file_size(url, ctx)
    if not remote_size:
        print(f"[ERROR] Khong lay duoc kich thuoc file {name}")
        return False

    print(f"[INFO] Kich thuoc chuan cua {name}: {round(remote_size/1024/1024, 2)} MB")

    chunk_size = 2 * 1024 * 1024  # 2MB chunks
    retry_count = 0
    max_retries = 30

    while retry_count < max_retries:
        current_size = os.path.getsize(dest) if os.path.exists(dest) else 0

        if current_size >= remote_size:
            print(f"[OK] {name} da tai ve day du 100% ({round(current_size/1024/1024, 2)} MB)!\n")
            return True

        if current_size > 0:
            print(f"[RESUME] Dang co {round(current_size/1024/1024, 2)} MB / {round(remote_size/1024/1024, 2)} MB ({(current_size/remote_size)*100:.1f}%). Tiep tuc tai tu byte {current_size}...")
        else:
            print(f"[START] Bat dau tai moi {name}...")

        headers = {'User-Agent': 'Mozilla/5.0'}
        if current_size > 0:
            headers['Range'] = f'bytes={current_size}-'

        req = urllib.request.Request(url, headers=headers)
        mode = 'ab' if current_size > 0 else 'wb'

        start_time = time.time()
        downloaded_session = 0
        last_log = start_time

        try:
            with urllib.request.urlopen(req, timeout=30, context=ctx) as resp, open(dest, mode) as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded_session += len(chunk)
                    now_total = current_size + downloaded_session

                    now = time.time()
                    if now - last_log >= 5.0 or now_total >= remote_size:
                        pct = (now_total / remote_size) * 100
                        spd = (downloaded_session / (now - start_time)) / (1024 * 1024) if (now - start_time) > 0 else 0
                        print(f"  -> {name}: {round(now_total/1024/1024, 1)}/{round(remote_size/1024/1024, 1)} MB ({pct:.1f}%) - Toc do: {spd:.2f} MB/s", flush=True)
                        last_log = now

            final_size = os.path.getsize(dest)
            if final_size >= remote_size:
                print(f"[COMPLETED] Tai xong {name} thanh cong!\n")
                return True
            else:
                print(f"[WARNING] Ket noi bi ngat giua chung o {round(final_size/1024/1024, 2)} MB. Se thu lai (lan {retry_count+1})...")
                retry_count += 1
                time.sleep(2)

        except Exception as e:
            print(f"[EXCEPTION] Mat ket noi: {e}. Dang thu lai tu dong sau 3 giay (lan {retry_count+1})...")
            retry_count += 1
            time.sleep(3)

    return False

def main():
    print("==================================================")
    print("RESUME & HOAN TAT BO DU LIEU CICDDoS2019 (SYN FLOOD)")
    print("==================================================\n")

    for item in FILES:
        ok = download_file_with_resume(item)
        if not ok:
            print(f">>> THAT BAI KHI TAI {item['name']}")
            sys.exit(1)

    print(">>> TOAN BO 2 TEP DU LIEU DA DUOC TAI VE DAY DU 100%!")

if __name__ == "__main__":
    main()
