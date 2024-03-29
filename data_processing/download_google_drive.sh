# NOT NEEDED ANYMORE. This script was initially intended to download many files from Google Drive onto a server. However Google Drive started 
# making this tricky, and now it's much easier to use rclone.

# To obtain the DOCUMENT_IDS, for each Google Drive file, click "Get sharable link". Make sure the
# permissions are set to "Anyone with the link can edit." Copy the link, eg:
# 1tYsp4aVW1bLD3gvwMdRxMG_xW2kdasjw
#
# Then: Delete everything except for the long number/letter string between "file/d/" and "/view". In this example:
# 1tYsp4aVW1bLD3gvwMdRxMG_xW2kdasjw


# 2018-08-01 Landsat
#DOCUMENT_IDS=("1rMAm92Ln84As26BT8fdy_Xb7YV-YXRdU" "1mGcruZV4VcoJPpWQ1456TZ3rUDDLUHJP" "1fCFB9NtwvcmU8F74rbvqf5HIyS6uiKrp" "1xhXT5GmwfwdZWZN48i0MnLZAtQlQQ7LH" "1d0DX7A3nGv905CsNS-xYlGXHo64nGdJz" "1tYsp4aVW1bLD3gvwMdRxMG_xW2kdasjw" "1r6Nmub8noHeFkeW1oyo8OIrg-acsl3rS" "1Z7oac_pU8VGD7lm1Pom-GznGUClm14JL" "1VpFJNXfSDI0jtmDS-Kcylr3j-_2PKDvh" "1Nw25X12ooAs65icUQX-zdO38d5k8VGSF" "1_4cXBS0JlIyKbZpgAzb2eyh8Qg4X9JZj" "11xgW274YggdVRow7E4igDJV3WSAjoESF" "1XQM20iaq4YzT8sZNOmZjeTLSK0fABfIM" "11lVQ5IsjhH8fccKiTd-qMi1EBE-VX51g" "1qrXZbHTSsaf6SW7MEQzfsOm80dNLS7fc" "1VAsl5NzYUaCRYqcnEA4mDC06KM_MmkBm" "1tgUMtypQfVvY_E29ZlmytLg7uFtK_E2b" "1A3U0bc88Y5M8H7d-BjrY1pRaKiS9I4Xs" "1EUHTVTuiYuR5mEQhfqqchXoqfSFjXX5S" "1f4_rY9IqSWShlUPF5__unQKpFqoqVrB6" "1BdVH511o5Tmt5lK4kCt1GzPN_PscCs93")


# 2018-04-29 Landsat
# DOCUMENT_IDS=("1IJwabsUE52zabA5thb2hYJ2GFXoXawlO" "1fysswa3a48q66kalFCVJmhv9lBI2-D47" "1B4CehJM46256qRUAHh4y4J5I6Dz1oLfd" "11sTTuhD0YVjOU-tra2eMDdxXU65qWdTv" "1gCymiZcrfPoTufN3o9ExDTcUSeHw-tf-" "1a9lkGvHTFyLEdKPqpoNoVEs-EKTfTu-K" "1TiVEmkc_b7AWiRJkZWZGnF_YiVtPs_kB" "1PD_VGtHavR_olxj2qgBVFZpP-J6TBk0O" "1OLF4LRJY7k9cpPVr2zNEkPOHI0Ie1riY" "1TuCIZF2S1FH8qk9KUC546mMZMoVZXd00" "18-Y84vDLvXSm1Y9tWgVw00RmgNpm0lKs" "1jrUtA1JS3MIWJnTSMkS_mZRAxnsBzz4o" "1W2zDXK2qGs0SL-NOn4P3wHz9OG549O-j" "1LSBXKVnQFChGV4WLe3oHexHNbAnzDHPO" "1-8j-_mnQxaIMsR4CUe6ijj8_Azl_g8TB" "1SGz1E7b1kXlrmHulW_LZT752AOjCLPs3" "1jC6fGARBQfSF4vmmeZGhyuzTsgLYqXmk" "1mkE4Nxd408wYad3z2Q46abI42kDZWWeO" "1SMMnU7F5ahMRKnjAukSNlbr3XXRO9wKT" "1Z4GkBbF9MdzMVHMLRZop_jtZT5vtXAOz" "1efjqu-njMykpAZLN20Wgg0gAqvcGMxsf")

# 2018-05-13 Landsat
# DOCUMENT_IDS=("1ilN_QaOfEUNr5UM9MoQoYZvPziH2fUJg" "1QhWmUCSSLj8sa55fCqNF5M7YB6jiMF2Y" "1R078KpbviBM9v5SjWvJrTX_6II8pZ_-q" "1egYk1K-Dc6CFOWctijIN8vla004nAhdS" "14cdgVJnn6TEdnzS11lTPWPZAg8m1vNf2" "1CXjx5ire9x6opCxHG62XNQTCWhz72neE" "1C8rIUgsGELkCI-LCY-MsBYnvAu6qQd5v" "1EDy8VbSTv10z8Fkk2eJ17uVFqVQrhs07" "1i21m4hYQwTQCEJVk13FVWsXV1dBU65im" "1bO9FMsUwaBwc-WCPc0neBoVOA50OxxLN" "1VKnFEqHldUcgZY9XurPXd9zTurKZkVTV" "1C97pgcqbbsWHuQ6Q8dAcdJrQrkMzNVwc" "1bXx-ExEzBCBAgSaxArpvyjM51MG1j7sD" "1MG-S8xzRDmf6r9OeIfm8IKT8OiUbx-n1" "1oUZH4uWVyBVkJuPGvsb9xuz_El0dHMDk" "1Ae7NYr8bTvMyF3jn-PlO7U3b7L3qPJK8" "1wpWtRQEjSuZke2ZsIG1jU5lqnljeV_ZJ" "1wO4VA8aWOrLQgrfFRSSs7tRHCAzKkhYK" "1W5IhQiY9WbiBAH9kZqON4kl3Fa2co5I8" "1YZiHRqfQgrDiXG0L2W5Mw4kcVf787LBp" "1qvxY3ghAlVHauAZmWISPQxIe8f_-c29L")

# 2018-05-27 Landsat
# DOCUMENT_IDS=("1fxvC5gGK5yFzIjIgZF6NRCyGaBjee_SR" "1iD64GMlVJkVclZH_gFEQM-Fvdb3ZAa8P" "1RkliOo6LeUm30fBRdjCm-ILcEZylPRyy" "15O6gHwnGyLr-Lo4PmbpEeml0GuxiMXLs" "1aDKvl9ONMEI_JWR52-2IvOcziJ5T9WNS" "1JQuCNWfKnhrDtH92lxoec0Xg_abVWYoQ" "18IFzQHwH-zqYuJKwFCJuX06r1ZhMvXXe" "1OEZnQngG2nPVn4DL4l2UVJofbUt-bL0r" "10wfJtKJuQiopLOfglr1FmV4VacdZgOr2" "1SSrVM2hsHD_EvKqWtno3Oot_EnfTzlVH" "1EA_zLo43OToJ-MH_2psgKg7xqXxAATtf" "1HVdwvHLSrlKA4UVkfZDUTR0CIiybj9r0" "1f2j7b6Z2ZCKGNBlI4WqZIU-86Vr_Q4lM" "1Ndy8vo1rGoDUtwNg2oK7iSZAINdEsa9_" "15h8TCHh5FSVpHE94oi6znT7O6qwtAmMU" "1D4m_WdaAtkWd7ShLYmfoCJDvxtRzs-hX" "11W5zU_KZh3Z5abgk_bwFh7EVTqYv-JCp" "1reMHgwE_l21Gv28vML8w9ksMq0vUAw_b" "1rzwoWOLMvXEUIH5biBlTEjdxJgVv08W8" "1g-yby9ackerJvugDQgdNc9t9v92Ii84z" "1u5vU8CRoM8RYROfTerXbuE-U_19dfmG4")

# 2018-06-10 Landsat
# DOCUMENT_IDS=("1j5733NXwyvGTaS_JuXcuK6djcKeqsCxZ" "14D-gu6R7LHx-6ZVHen24LO4iwSPkUnWH" "1Qy7_Zbg6XlHDWoHxwpugJ8I78CZbYQ0O" "1qyreM1uJ2dcv62mu4-e-wk00G4ozcshm" "1jylEpfOYyFHxPz-eaz6Uyd6BHZDEb5Ma" "1Gc-wXIc6i7M6eIR8QkI3VWvHyNbSJ0Gq" "11khC8jCoZamTwSlEP9NnYufwRg5pugbn" "1mYNKdObWqVwuFchyvItZcCPpWt-wS2Se" "189wZt6kC2WCUUHdpxiKJI09cvwInkNrx" "1M-nbvsJy4Yhjbii6XklOYSUuoarcn-v-" "17TZymaPWOb-pW5Pfh2d4F7hkRkHm6i6T" "15dOyyHekFQWrkNAoy6p8mtKaHoENXA86" "1eYD-Nd3ik2IUbh1SdI3MY2gyT7TH2e_v" "1rDVmnSJCrCXGoqoKmI-_cAc91PE_b_qC" "15DsLnhbm-Bd_NjFDx5mWYsfUabXfQ60M" "1UeYBueD38BF878Bfq-EtZU-25jb8coKt" "1YbeOi9842ZD-e_arqs4YS6-ymsBY_9QC" "11tExAtt8VocLXiTcWF5jPo79KattyLF8" "1kavAeSv0u1wgAlgrAcY8s-Cua8-EEoWl" "1Dr2r8KVwW86vmifTmF-xMuSoOwIuk-Nh" "1Fb0OXy2Qr3TtR5gQ6RacCv3ru2fZiLI_")

# 2018-06-24 Landsat
# DOCUMENT_IDS=("1bvjdfw2rd6IV-w0kZUKaKiQmQyOdDZrh" "1c5nBWc8HfRvs9d_qB2T1zo6L-W8zxdvX" "1ORJScRC-Jp5dpUe7AlwN-l5hUAB-4YZ_" "167dD4ClnZg9GAjSXZzrHSGyfT6kaDsLQ" "1a5uWPqaV7wkJiUhFr9HbC85C72KZw5CL" "197ECS_l-zLUe00UvVqhWZs6VTcAK4uui" "1dfaI1_Gv9RWkPWyi6LzQ2LBcfbAKF4To" "1Y1gKxuunSO_knsKVpYVOvHfbzCtFVZxc" "1udaWi2OcB7D4DNB2RwMl7kV0prVm8tqN" "1LGf6g5NMbzU1gytK2DRGgwx08yEixW6l" "1wH9Iq4FmJLQG5PWKkHuz6ULwEEhVXiHG" "1tADP7MoDwaCKLOHk4mpkL5ux7vWPOqPv" "1YJmBhu87UMDevcV91q6XgA9IiF0jJOcV" "1A-G0vZwpsC4RV62V8Nq7-OFKdKhSGYnP" "1GO-eRo_9LxjnM4MbPw6d8XMX2B0OG__B" "1qc183eM8S_EOAJ3LhwZdRL_YUPn9Bf2C" "1DWWZlpFdWXBTglEQnN8vEwBtx02RCey-" "10h2pt9gvhTRDR4LdOD4k_tg-YsTl6YrD" "1_nhcJwhARxXzlEr7IIbe5Tq58yHkT3yn" "1YEhiaDWSgsg9QVdzKxiF2kDJC17qjb0m" "12eDbc_xCFCB1S2Ga43zVnkFnQiZ5WQDF")

# 2018-07-08 Landsat
# DOCUMENT_IDS=("1tMYVNfepg9rUA4vPJltGL2v17xtZzHH7" "1TBDt9xr6t-ZEYShZwTBZbdR-Ux1RbZEU" "1Wk3yfy4qHkBdAWOq6WStXiYpsNIlQPYt" "1ybyMTVOsNO2SOMwglSNZ1EK4iTRzCpFP" "1iqRS_N8TNB6U1_9SZrzxPjlnlTteSITz" "19lg9YC6ry96NBQvj25L0sR8ZW1p8V6Gy" "1beEJkqYY9b3dzeUjCgHUni2CtI3N9eo7" "1NqR9r73iODa4MAcBXMzuPp94BM-hpNgz" "1zAvJcJS3xXN5hIE8N7ZqjVB8UdmpWDH3" "1OkZGXLdEAOI9UNz9M8s49NnG6HL80Ccy" "1TlLp2WsybGEVJ0LObGTVP5D-8wr7A7a4" "1TWaxqbzNXneiCqxFerNwRKwWzBnhDcay" "1qiDpZUAtt63xYqXhDntjbY1QJSmJ15C4" "102cW-m_ecNt1s2Zfs57H7gMgi4J7YypJ" "1PkgvK6-Boi-I7i2JFEKiss5n-Eanh5ot" "1EWNcUcNQrIBdjGlynx3oZc9KIqiMIDoP" "14AMGJuQRjcGITFUEdL082MzfVzOOFdQ8" "1UsAcJKGqQgjCC2DVBfjAE8vfQKI-sMJq" "1uPakZ0xn8HZsvbSWo86czrZpNmzDXW8T" "1eYb2cgCyzbN81VRos6jq2iZyLeXQK0w7" "1DyyQkW9m3eJvCgEqaQwRRNiVPpmEEr5j")

# 2018-07-22 Landsat
#DOCUMENT_IDS=("1yQNvyTYleHt2QQp4QQBVFmF9GTMxZoem" "1bM9kSmt5Xe4AwgDPK1sn1wpd9cFySE_a" "1P2F2WzA4sZZ_Z5hqYSrce4pwp7ZvK5Qw" "1eb8nCQrHdhL5H0CKbCzNBRYzk4grFz4A" "1vjnfGMHBNHc61zEXDY0XPctl77esPPuz" "1juonalA-WQ9Ft8b6hyZqnpNPV_S-JsIc" "1LaLS6hyIi1qRKVJYuarfFRgKXAXQ-DWR" "1dAlaeW_mPiyznLranKiWac0gfiV18T9C" "1ilgGbFFJb6hapxjZk5QDDutnM6tEFcvS" "1ZeR2nbPkswCbZwKo2BMe83aL2GfNx8uW" "1jOHhvu_FkuyoBLEy_25XS7dGk7ZJWI7J" "1G7YQ_AECW1qJWmHysAZAm-4NyyVObDQB" "1Z-2ng0Wsrsgp30DA9uL6Yq8nuJtnjR1n" "1BJyfuyvXKTEwvDK7Mryn4hcXRy1mAP-i" "1Ww9eeH4KMTWSoW5v6F-MflPLGhOg8r8V" "1LGwezwDOgJSt04dFYE1P3vXq2DiBl3Tj" "1WDZn0h-xlm4GOYgsVwmfUK7mCsUqnaDg" "1N24rGYwvZUd9xHICOYxe3vhRJzOm7PME" "1GltQlWIYQgQE7pKnB8UAZfbgYvDpvE5g" "145LIwevMNOvitqQq7c579nmwLh5FFDfA" "19nUlU8IyB-XQc0iCjSC9fZz1sJQr7sXL")

# 2018-08-05 Landsat
# DOCUMENT_IDS=("1jwQogR0EQP-vYVHMf3AKf3s8nDCUrvqX" "16EfyWzGd58orTiSHHYK73-FjEmpAgXt4" "1WPuYqRsZM--wnn5dcviawZWR9Qf3oHxw" "19Vux1PiMOQOBjrVASDX-QjX8zNV30Jmz" "1vq5y6BbxcyHEsPzZEnN244k_zAjgOVq8" "19nwRSWseVK1Sp3Pl1Xw89Sv_0I-NZGmg" "1x5B_Gh8B2272Px6DWc00Ju7f5qQdbXhg" "1w53RiI9e0-LRG6Fv_p2Hk_uHE0FHDYmb" "1KmrCKMqiJ6etOO079qUQ4PiKfA2HNBk_" "1fqnxzgR3cNSHzC6gsptPXbhTPTVZtYo7" "1v0a69QVfAGwbT70NwkqYIihI7bwdJ6FG" "11HHLcu-kWidWVrfxlYZ62R8tWsNJIlqF" "1YklGTC3VwjTsd_MtRV3xlxlcvxWY_lz5" "1OpLE4eKlFKaSRkNAD0_3gjK_bySP1YC7" "1mDrSSCF5_HhVh6SV_gZg6m3XDwToynVD" "1vYbKzZAQpuVAhhwZbKkbqKOu5TnPKWQ6" "1aUz3sLcVPEuQXCfbYdyjz1mEY3pp-Xy1" "1irRshQWleJPW-FdNl9v2Rd8b_hJkzu1U" "1tCSxlGI0wYBzhdy3L8NzTu6vpnPYAvTf" "1_yFcaWmajl3dL9PXFc6R7FXzCZpMC7gK" "1ivHjaeJCj14BAAP4IYUkGHVUyMfLbuWY")

# 2018-08-19 Landsat
#DOCUMENT_IDS=("1AaQoYrPkdNQS3sp-FQboq8ELtn-d1fn7" "1kQmo1ICVFESkhNIz-YjrMvmeBjANznqt" "1l4BRKqy8SZ83QMt60SbHjIBbkOLYO65l" "1Kuv6v7EJqAO3xE7qPnKc7PMjGOyaGFYx" "1EFU1hyXqIV5Hb4i0T62p4_3NM5OqZWdR" "1zoG3ewmbKlhV-zx_a4wWSxqSDDwZvMEB" "1UY_dbmO7pA5A7DqB-nVSGWiR9XamCqJH" "14HprEMrlqmy_HsUTDmUHZDWlWi33uuTd" "1R6btZztKgHtasONmUqG9W4gr7uy7DTFY" "1sJ2mkV1IPdQTNnqHfG9UP5if0NwAZ-pc" "1s3N-DZA4ekLUVszJPnS4wnOtpH-yMYMY" "1VJbxhUgSDzqg_3KlGGKi8S_Mqz0nHPQc" "1-Lytf4XR-Z0LQWD5AWPfWk-B5U-O6LGF" "1df7vekqG3ZhxdA4Pq4wQQEoMPIGrAt0S" "1-pIJ8VgXk3DBtgaTJ8pOlfo1R0ynM1lY" "1cWzD42HTG-qZUmxqHw91PXJskY7ncdp-" "1woEW0Md0CI6o9PwYzBivyRG59LJzUhnR" "1jx34q-T_T9Kvp08VWsQgmN4fwM7WYMgl" "1hHcr3HPdBwzr-9vMSYNvqiakbiYeEI-V" "1ecKoC_hc8uP7D6T9SomiLG4Z71ARDuSs" "1wWmLpmCR_0Dn7dGIphRT0Bnl0giTnmwH")

# 2018-09-02 Landsat
# DOCUMENT_IDS=("1jEzN0R9rN8RbZ9ry2JpXd7koiQy0c2nh" "1gMlVAvoCEVQHehwfRUVcQZiXPJUmV70Z" "1-N3hIQSO9ouXmlP9PESTSyJr8AaO9iVv" "1IXhZiiwhvKAWF3IxomH5mKJIFdhNytfk" "107V2m662ALY6p7gQEWtQiYJhRqp7C5kR" "147ogKVvT78Tqs4SvKF5c7iSPtve-CiQE" "11VxPmJytunfC8AJqW6Y9oENrXCaTcytB" "1cRstyZTOVRTZ6yVW64xZ76bgjPTFl3bj" "1XJSc1ttFNFF-8aMdjIdZFxMKFkxocl-x" "1CVokc7xGtBF2EOaM-deP8d_PsY5FS6_u" "1ECLj2TdiUTVl2juQo9iBUb2cc2IggoDx" "1n815ypGP6bNpA6EPU2i1xnuogYWiXASj" "1Cs07xAtlBv-EHKxuI7KbxHOhiWdEHZJc" "1dPJbHtP03H1UcHhZtSwapLZ6jCfmXPmz" "1wEEsqiz5BjR3n9RytMWENsNpIYdJZ70x" "1w3EUj947vODxP7KrRGJuYPER-xdY7VFe" "1RLLSwJWOcbkQpcTvuSyjzOEjfaEiHIMP" "1i1efg9EAOHpWpqkW7_b24pPEdijOjG_o" "1eFnKlwazeqR02UW9rSSjamfNNSUaOZ4N" "1u_UmEP-JNzgpDCd092AIhgEj9pV7c0ls" "1BFfUkX8TDek2YLMG1sC-2oMoCd9n1mbl")

# 2018-09-16 Landsat
# DOCUMENT_IDS=("1MAFA-x_MHqxYaglTwcVH1-yZ-4HORWly" "1SRCtKhSgSoCCL0D0P6hqeeO3dsFpJqpO" "1Pfe8ZkAatrMs_LBhMBGKJuKR-aVieM2p" "1u5kQD5qoLTi-NF-vwklwq6ev8XuXnntp" "1-kEolPGmbGJFyw3HHipW0kuzJrqZCaxj" "19tLbQrH2gLnQ8HLiXxxJHIvgEHLzO8yL" "1nJ1YYpnhD-BFmFrthiudLVgYBg6tTE9Z" "1cNkQVzyK5IKMd9GzstZCFNLfyJhLt8_5" "13ex_n_5OFZ8RxYJwhLtG5SqTDdaClrM5" "1fJRz11Fcj28UAq19iA3EiTvWj4L5WXGq" "1-uzUi3mddXChaU5NrcfVumxjOCNCBL4O" "17OUDitW_WN2CEK4KTVBBIsrBjJK2D5Ql" "1g3nhN8quEjqkZBKlidxrbd_XtqnuoG_e" "1phbnoeAEd9kinaxKU_qtJm-7AyeoMpzd" "1qOZJR6DcyKKsqVkGJ_Joq5SjfT5oQYi6" "1jVjNBZi8pz0Q4vtePejBPVYHiIdIqpkv" "1C56JYHN4pv5fzJsQpkdoN7EZe3VKEVV3" "12DtWbEs8MZkTvdZ0tyJcAHF65o4jb2eK" "1eOPA8sQ-D84wqwltG7-RT6WYmTcv2aHH" "1bOyeFJeZFhBk5UIJ_5Ppbu4CfMZYcg4s" "1pylUcCPPpCUUVmebygUIBJ0P6uhv1DrJ")

# 2018 CDL
#DOCUMENT_IDS=("1_XR322SooLPsPpy47sBlPcdfNL_dMytz" "1ajjhIFBDv79gQ_4f98tn-Yqq1Qtxw-T8")

# 2016-06-15 Landsat
DOCUMENT_IDS=("19Fb_A9FbvYGyX_llSIuUlH__3fFLZqqR" "1PEFR_a8KMUfcfT5uEV-EuotWbNDqeziZ" "1SuCN0mPuu81FcZJabdSfcX6KeA8t66el" "1FI7T881lGVPJFB9KvBc61IRX7EnKHpjJ" "1LnxISeDtIeHpvYMVswYxbKfLCz-K_r7B" "1ybPedF-a4rIPWLZL9yz1CwGQ9S_vhCCY" "1F0TxwBoGnaZ4OcLQYtcNbn2SqpQZbbK2" "12GPzR--0IECN26W2dumudHj_-lv3CRus" "1p3hxM1rmff31JkzelKljhFLmL6jPZhtf" "1mA8AT1XtL3_QHAinElxrIUhSV31L5NKx" "17sWB47E4diCNa4v1yVHVbCRR5OXcpvOD" "18JrXBJTJZzkHaKIZNBSFnPuHNvjzwFsw" "1FOyeaTV4-vA0E4l1qr2kxFs3H1U4cYIx" "17WecOPrtj7c7yig6lsSusxvBS9CzFfuF" "1Gblwvn0y2YsTQBIny84dtOYXuhehfOUW" "10h3tH35NxXh2KN5eP263BpO-nhuHdL7o" "1lYaMtOtrwpfmt8PjBFtRVEduH5odYo5j" "17wzQ67jKo5vRkOwEcJ0rTVPEW4__xddd" "19V0356ae21pGU2dxS_GIXDGN8MsOIwjk" "129v-Ip5B3CY6b2gKxrFzFm8fUqaME4kj" "1aKwdD9Fua_oVINbUjLrl7WHpH9r-tE47")

# 2016-08-01 Landsat
# DOCUMENT_IDS=("1jdyeJNfE7MReOp8hNvTksLa6Jw1bzlzT" "1yYtiVNE1KV4_wl_i8DFHcy-ZhLFaoKML" "1GaCMaDe_6gfPd7ejLlCWPDdFLaaN2us2" "1ru_j5b1sKCZ4_cGPYtlq6zQbVW8GmyCB" "16BcfzCdjrZ5qiESrtQ8Fn46RAVkMI8Aq" "18-pmY_084KA2zpiUqiQbFuLalDMZqYLo" "1m21sjzhehaf728QaF8mXJ_gFN5p8ga6L" "1I3Z-6HGgj7TYcfv7cLGiPf5djKHzOH1e" "1DO5k_bYXMDew5EB6G1GpockT6sdAB5k2" "1KTCy4e76LmOSjJIvaLc5oDJp-9fZBsxw" "1vs15E4vS0H25JsiiQBplvV1gpGO6jyVo" "1njI-BM7jHAFeh8ws0oxnyVyfd8ZzUF65" "1UeECIUM8WMI2m-rZSOA4okFP2ZijHVId" "1D0cGqABj4ow6owMVzGW94UYJw2AHTcEh" "1_eUVzllYRIa9SkrLnUNSELBq0m3eLl0b" "12eRO5rnvrV77IEDJtguwMLUPO7UIhofV" "1hPqQ2RcOa6h1pC2EE1jVTpe2W3xQDZ3r" "1-hHG3R8na9VmT4rMryN_Tlv0_-VLdlAO" "16ySGbT_-P2v-cSZmoSCEQqAnKoXg5MX7" "181uqEUs5I37pLRlLFHulV_Xm8lVoCUek" "18X4NkeoBcvBRJJZau3LUbQhdoqgtnDts")

# 2016 CDL
DOCUMENT_IDS=("1fTCPDaQhQH1mkwlqUfcvEJhszVqKUaxI" "16PjxBRu8ANuScfH_GzR9HdOQLNcxh4mc")

# DOWNLOAD_FOLDER="/mnt/beegfs/bulk/mirror/jyf6/datasets/LandsatReflectance/2016-06-15"
DOWNLOAD_FOLDER="/mnt/beegfs/bulk/mirror/jyf6/datasets/LandsatReflectance/2016-06-15"

mkdir ./tmp
mkdir ${DOWNLOAD_FOLDER}

for i in ${!DOCUMENT_IDS[@]}; do
  x=$((${i} / 7))
  y=$((${i} % 7))
  FINAL_DOWNLOADED_FILENAME="${DOWNLOAD_FOLDER}/corn_belt_reflectance_box_${x}_${y}.tif"

  curl -sc ./tmp/cookie "https://drive.google.com/uc?export=download&id=${DOCUMENT_IDS[${i}]}" > /dev/null
  code="$(awk '/_warning_/ {print $NF}' ./tmp/cookie)"  
  curl -Lb ./tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${DOCUMENT_IDS[${i}]}" -o ${FINAL_DOWNLOADED_FILENAME}
done
