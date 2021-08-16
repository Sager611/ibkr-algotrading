#!/bin/sh

# exit on error
set -e

# create directory if it does not exist
[ -d container_inputs/ ] || mkdir container_inputs

# check if there are already certificates
{ [ -f container_inputs/cacert.jks ] || [ -f container_inputs/cacert.pem ]; } \
    && {
    echo 'Certificates already exist in container_inputs/. Do you want to remove them? [y/n]'
    read -r OPT
    if [ "$OPT" != 'y' ]; then
        exit 0
    fi
}

# remove existing certificates in directory
rm -f cacert.*

# kindly taken from stack exchange
LC_ALL=C
PW="$(tr -dc 'A-Za-z0-9!#$%&()*+,-.:;<=>?@^_~' </dev/urandom | head -c 30)"
echo "Generated random password : $PW"

# RFC 2818 states:
#   If a subjectAltName extension of type dNSName is present, that MUST
#  be used as the identity. Otherwise, the (most specific) Common Name
#  field in the Subject field of the certificate MUST be used. Although
#  the use of the Common Name is existing practice, it is deprecated and
#  Certification Authorities are encouraged to use the dNSName instead.
#
# So, we provide the SAN (Subject Alt Name) extension for localhost.
printf '\n' \
    | keytool -genkey -keyalg RSA -alias selfsigned -keystore cacert.jks \
        -storepass "$PW" -validity 730 -keysize 2048 \
        -dname 'CN=localhost' \
        -ext SAN=DNS:localhost

printf '%s\n%s\n%s\n' "$PW" "$PW" "$PW" \
    | keytool -importkeystore -srckeystore cacert.jks -destkeystore cacert.p12 \
        -srcstoretype jks -deststoretype pkcs12

openssl pkcs12 -in cacert.p12 -out cacert.pem -passin pass:"$PW" -passout pass:"$PW"

rm cacert.p12
mv -f cacert.* container_inputs/

# replace password option
sed -i "/^sslPwd: \".*\"/c\\sslPwd: \"$PW\"" container_inputs/conf.yaml

echo Done
exit 0
