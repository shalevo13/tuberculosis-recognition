"""
Utility to export web/index.html into an optimized C header for ESP32 firmware.
Generates both raw minified PROGMEM string and gzipped byte array.
"""

import gzip
import os
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML_PATH = os.path.join(PROJECT_ROOT, "web", "index.html")
HEADER_PATH = os.path.join(PROJECT_ROOT, "models", "tinytb_net_web_page.h")


def minify_html(html: str) -> str:
    # Remove HTML comments
    html = re.sub(r"<!--(.*?)-->", "", html, flags=re.DOTALL)
    # Collapse multiple whitespaces and newlines
    html = re.sub(r"\s+", " ", html)
    # Remove space around tags
    html = re.sub(r">\s+<", "><", html)
    return html.strip()


def export_header():
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        raw_html = f.read()

    minified = minify_html(raw_html)
    gz_bytes = gzip.compress(minified.encode("utf-8"), compresslevel=9)

    print(f"Original HTML size: {len(raw_html):,} bytes")
    print(f"Minified HTML size: {len(minified):,} bytes")
    print(f"Gzipped HTML size:  {len(gz_bytes):,} bytes ({len(gz_bytes)/1024:.2f} KB)")

    hex_array = ", ".join(f"0x{b:02x}" for b in gz_bytes)
    # Wrap lines every 16 bytes for clean C formatting
    hex_lines = []
    chunk_size = 16
    for i in range(0, len(gz_bytes), chunk_size):
        chunk = gz_bytes[i : i + chunk_size]
        hex_lines.append("    " + ", ".join(f"0x{b:02x}" for b in chunk))
    formatted_hex = ",\n".join(hex_lines)

    c_header_content = f"""/**
 * ==============================================================================
 * TinyTB-Net Embedded Web Page Header for ESP32 / ESP32-S3 Firmware
 * Auto-generated from web/index.html
 * Gzipped payload size: {len(gz_bytes)} bytes ({len(gz_bytes)/1024:.2f} KB)
 * ==============================================================================
 */

#ifndef TINYTB_NET_WEB_PAGE_H
#define TINYTB_NET_WEB_PAGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {{
#endif

// Length of gzipped index.html asset
#define TINYTB_INDEX_HTML_GZ_LEN {len(gz_bytes)}

// Gzipped HTML asset (Serve with Content-Encoding: gzip)
static const uint8_t TINYTB_INDEX_HTML_GZ[{len(gz_bytes)}] = {{
{formatted_hex}
}};

/**
 * ESP-IDF / esp_http_server Handler Example:
 * 
 * static esp_err_t index_get_handler(httpd_req_t *req) {{
 *     httpd_resp_set_type(req, "text/html");
 *     httpd_resp_set_hdr(req, "Content-Encoding", "gzip");
 *     return httpd_resp_send(req, (const char *)TINYTB_INDEX_HTML_GZ, TINYTB_INDEX_HTML_GZ_LEN);
 * }}
 * 
 * static esp_err_t predict_post_handler(httpd_req_t *req) {{
 *     // Expects 50,176 bytes (224 x 224 grayscale) sent from client canvas
 *     uint8_t input_buf[50176];
 *     int ret = httpd_req_recv(req, (char *)input_buf, sizeof(input_buf));
 *     if (ret != sizeof(input_buf)) return ESP_FAIL;
 *     
 *     // Run inference via ESP-NN / TFLite Micro...
 *     float prob = run_tinytb_inference(input_buf);
 *     char resp[128];
 *     snprintf(resp, sizeof(resp), "{{\\"probability\\":%.4f,\\"is_tuberculosis\\":%s}}",
 *              prob, (prob >= 0.35f) ? "true" : "false");
 *     httpd_resp_set_type(req, "application/json");
 *     return httpd_resp_send(req, resp, HTTPD_RESP_USE_STRLEN);
 * }}
 */

#ifdef __cplusplus
}}
#endif

#endif // TINYTB_NET_WEB_PAGE_H
"""

    with open(HEADER_PATH, "w", encoding="utf-8") as f:
        f.write(c_header_content)

    print(f"Exported embedded web header to: {HEADER_PATH}")


if __name__ == "__main__":
    export_header()
