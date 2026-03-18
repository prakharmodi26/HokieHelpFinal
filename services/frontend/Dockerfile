FROM nginx:1.25-alpine

LABEL org.opencontainers.image.source=https://github.com/prakharmodi26/HokieHelpFinal

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY index.html /usr/share/nginx/html/
COPY style.css /usr/share/nginx/html/
COPY app.js /usr/share/nginx/html/

EXPOSE 8080
