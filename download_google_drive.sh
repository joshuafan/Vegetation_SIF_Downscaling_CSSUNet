# Aug 16 2018
DOCUMENT_IDS=("1V68AUKVQXGS2ahNb1VEYkcUdF28KnLxc" "1ITiRX8-Xr754rvSrURgJcHD-kgJQB6w8" "1OjT77lk9Y1BMMd6BBMDnllgnOFSYg86J" "1bd51LkhnYMjMgxMfAtsbjy9jjDaVOmlG" "1jvJ-vQlI0sEdUw1mooOieU8SoIE3b4c4" "1BLHoeArhxRg047Vo_brB734NXIOVY60J" "1mlD1mz7E3QYPMRMmYAReHetRV3gkv7w1" "16chY8muaqfNBpMGnJoeFPdOZTdwjIYeN" "1XGCEzUf6RrHBSIHcrAeKpE8O8_hrUypJ" "1KP_7ob3DYN9hBW8wLudY6ReIxUB-CJsE" "1hqu2B0aYabcO3tSctci4uPkSvi_yug-Q" "1df1jHxen1daqtOZT6qzaDCZvrlEf-6-t" "1zZ_v9HzrVlRgsu8Ze4gXgx3cujMtKIWl" "11aDQgbav1JXkN8TuDjBmpQZRwzJoGORK" "13tZPfHAvykBZH4fs2tHNjRuOdU6vv2qD")

# Jul 1 2019
#DOCUMENT_IDS=("1DpjsKeGj4bd03WXh7cK1xAlf_OSksTQc" "1sLHRam094qycsPc7_sL9wtogiVnLjFAR" "1bjYtLzTwHqWyds446RvGi0Y_3uRXyZyC" "1zVs-cdlt49kU7vdF5VW8Hkyp7-1UpLb8" "1dL2hCznLVfGYxrilBvJaGwF9PbA1DvuI" "1FPPv-F7pk4ySEQfw8lKlv31pz40gjz3H" "1ILo_PxxWhPXVJ_EOn_POKCqWqNnqDK83" "1Om7hEQwxOMkIStSi7l7D3vN5ekL2iCz9" "11cB3p8rIyNakAoxXJaV_CKVJdNcU40gW" "1QeGQGjGL6HMjWPkjDCeyvbADudN0fu8T" "15MOI_QDiCNn84-x4vAcT5-6cruJy798z" "1lAL3_b10kwvIngeSQ4wIUkihBt-cCBnB" "1gC69DSJ-phbpM0SgXWQfa7svPUVzx3DE" "1j8xlbxOzBGSp6WgAWag2KXXrkylVpEXc" "1ZdBOgWy_IlaAhz4oQFK_LVlne5ELblbG")
mkdir ./tmp

for i in ${!DOCUMENT_IDS[@]}; do
  x=$((${i} / 5))
  y=$((${i} % 5))
  FINAL_DOWNLOADED_FILENAME="datasets/LandsatReflectance/aug_16/corn_belt_reflectance_aug_16_31_2018_box_${x}_${y}.tif"
  #wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=${DOCUMENT_IDS[${i}]}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${DOCUMENT_IDS[${i}]}" -O ${FINAL_DOWNLOADED_FILENAME} && rm -rf /tmp/cookies.txt
  curl -c ./tmp/cookies "https://drive.google.com/uc?export=download&id=${DOCUMENT_IDS[${i}]}" > "./tmp/intermezzo_${i}"
  curl -L -b ./tmp/cookies "https://drive.google.com$(cat ./tmp/intermezzo_${i} | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > ${FINAL_DOWNLOADED_FILENAME}
done
