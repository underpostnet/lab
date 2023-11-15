import fs from 'fs-extra';
import AdmZip from 'adm-zip';
import { Downloader } from '../src/server/downloader.js';
import { getRootDirectory, shellCd, shellExec } from '../src/server/process.js';
import { loggerFactory } from '../src/server/logger.js';
import { s4 } from '../src/client/components/core/CommonJs.js';

const logger = loggerFactory(import.meta);

logger.info('argv', process.argv);

const [exe, dir, os, program, hostPath = ''] = process.argv;
const [host, path = ''] = hostPath.split('/');

try {
  let cmd;
  switch (program) {
    case 'certbot':
      switch (os) {
        case 'windows':
          await (async () => {
            const urlDownload =
              'https://github.com/certbot/certbot/releases/latest/download/certbot-beta-installer-win_amd64_signed.exe';
            const folderPath = `./engine-private/setup`;
            if (!fs.existsSync(folderPath)) fs.mkdirSync(folderPath, { recursive: true });
            const fullPath = `${folderPath}/${urlDownload.split('/').pop()}`;
            logger.info('destination', fullPath);
            if (!fs.existsSync(fullPath)) await Downloader(urlDownload, fullPath);
            cmd = `PowerShell Start-Process -FilePath "${fullPath}" -ArgumentList "/S" -Wait`;
            shellExec(cmd);
          })();
          break;
        default:
          throw new Error(`Os not found: ${os} for program ${program}`);
      }
      break;
    case 'xampp':
      switch (os) {
        case 'windows':
          await (async () => {
            const versions = {
              '7.4.13-0':
                'https://ufpr.dl.sourceforge.net/project/xampp/XAMPP%20Windows/7.4.13/xampp-windows-x64-7.4.13-0-VC15-installer.exe',
              '7.4.13-1':
                'https://ufpr.dl.sourceforge.net/project/xampp/XAMPP%20Windows/7.4.13/xampp-portable-windows-x64-7.4.13-1-VC15-installer.exe',
              '8.0.28':
                'https://sitsa.dl.sourceforge.net/project/xampp/XAMPP%20Windows/8.0.28/xampp-windows-x64-8.0.28-0-VS16-installer.exe',
            };
            const urlDownload = versions['7.4.13-0'];
            const folderPath = `./engine-private/setup`;
            if (!fs.existsSync(folderPath)) fs.mkdirSync(folderPath, { recursive: true });
            const fullPath = `${folderPath}/${urlDownload.split('/').pop()}`;
            logger.info('destination', fullPath);
            if (!fs.existsSync(fullPath)) await Downloader(urlDownload, fullPath);
            cmd = `${getRootDirectory()}${fullPath.slice(1)} --mode unattended`;
            if (!fs.existsSync(`C:/xampp/apache`)) shellExec(cmd);
            fs.writeFileSync(
              `C:/xampp/apache/conf/httpd.template.conf`,
              fs.readFileSync(`C:/xampp/apache/conf/httpd.conf`, 'utf8'),
              'utf8'
            );
            fs.writeFileSync(
              `C:/xampp/apache/conf/extra/httpd-ssl.template.conf`,
              fs.readFileSync(`C:/xampp/apache/conf/extra/httpd-ssl.conf`, 'utf8'),
              'utf8'
            );
            fs.writeFileSync(`C:/xampp/.gitignore`, `/htdocs`, 'utf8');
            shellCd(`c:/xampp`);
            shellExec(`git init && git add . && git commit -m "update"`);
            shellCd(getRootDirectory());
          })();
          break;
        default:
          throw new Error(`Os not found: ${os} for program ${program}`);
      }
      break;
    case 'docker':
      switch (os) {
        case 'windows':
          await (async () => {
            const urlDownload = `https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe`;
            const folderPath = `./engine-private/setup`;
            if (!fs.existsSync(folderPath)) fs.mkdirSync(folderPath, { recursive: true });
            const fullPath = `${folderPath}/${urlDownload.split('/').pop()}`;
            logger.info('destination', fullPath);
            if (!fs.existsSync(fullPath)) await Downloader(urlDownload, fullPath);
            logger.warn('You must have WSL, install with the following command', `wsl --install`);
            cmd = `${getRootDirectory()}${fullPath.slice(1)} install --quiet --accept-license`;
            if (!fs.existsSync(`C:/Program Files/Docker/Docker`)) shellExec(cmd);
          })();
          break;

        default:
          throw new Error(`Os not found: ${os} for program ${program}`);
      }
      break;
    case 'vs-extensions':
      fs.readFileSync(`./vs-extensions.txt`, 'utf8')
        .split(`\n`) // \r\n
        .map((extension) => {
          if (extension) shellExec(`code --install-extension ${extension}`);
        });
      break;
    case 'wordpress':
      await (async () => {
        const urlDownload = `https://wordpress.org/latest.zip`;
        const folderPath = `./engine-private/setup`;
        if (!fs.existsSync(folderPath)) fs.mkdirSync(folderPath, { recursive: true });
        const fullPath = `${folderPath}/${urlDownload.split('/').pop()}`;
        logger.info('destination', fullPath);
        if (!fs.existsSync(fullPath)) await Downloader(urlDownload, fullPath);
        const confServer = JSON.parse(fs.readFileSync(`./engine-private/conf/conf.server.private.json`, 'utf8'));
        const { directory } = confServer[host][`/${path}`];
        logger.info('client found', confServer[host][`/${path}`]);
        const zipTargetPath = directory ? directory : `./public/${path ? `${host}/${path}` : host}`;
        if (!fs.existsSync(zipTargetPath)) fs.mkdirSync(zipTargetPath, { recursive: true });
        const tmpFolderExtract = s4() + s4();
        fs.mkdirSync(`./public/${tmpFolderExtract}`, { recursive: true });
        logger.info(`Please wait, zip extract to`, zipTargetPath);
        new AdmZip(fullPath).extractAllTo(/*target path*/ `./public/${tmpFolderExtract}`, /*overwrite*/ true);
        fs.moveSync(`./public/${tmpFolderExtract}/wordpress`, zipTargetPath, { overwrite: true });
        // https://wordpress.org/documentation/article/htaccess/
        // https://github.com/chimurai/http-proxy-middleware/issues/785
        let htaccess = ``;
        if (host.split('.')[2]) {
          // sub domain
          htaccess += `# BEGIN WordPress Multisite
# Using subdomain network type: https://wordpress.org/documentation/article/htaccess/#multisite

RewriteEngine On
RewriteRule .* - [E=HTTP_AUTHORIZATION:%{HTTP:Authorization}]
RewriteBase /
RewriteRule ^index\.php$ - [L]

# add a trailing slash to /wp-admin
RewriteRule ^wp-admin$ wp-admin/ [R=301,L]

RewriteCond %{REQUEST_FILENAME} -f [OR]
RewriteCond %{REQUEST_FILENAME} -d
RewriteRule ^ - [L]
RewriteRule ^(wp-(content|admin|includes).*) $1 [L]
RewriteRule ^(.*\.php)$ $1 [L]
RewriteRule . index.php [L]

# END WordPress Multisite
`;
        } else if (path) {
          // sub folder
          htaccess += `# BEGIN WordPress Multisite
# Using subfolder network type: https://wordpress.org/documentation/article/htaccess/#multisite

RewriteEngine On
RewriteRule .* - [E=HTTP_AUTHORIZATION:%{HTTP:Authorization}]
RewriteBase /
RewriteRule ^index\.php$ - [L]

# add a trailing slash to /wp-admin
RewriteRule ^([_0-9a-zA-Z-]+/)?wp-admin$ $1wp-admin/ [R=301,L]

RewriteCond %{REQUEST_FILENAME} -f [OR]
RewriteCond %{REQUEST_FILENAME} -d
RewriteRule ^ - [L]
RewriteRule ^([_0-9a-zA-Z-]+/)?(wp-(content|admin|includes).*) $2 [L]
RewriteRule ^([_0-9a-zA-Z-]+/)?(.*\.php)$ $2 [L]
RewriteRule . index.php [L]

# END WordPress Multisite
`;
        } else {
          // main domain
          htaccess += `
# BEGIN WordPress

RewriteEngine On
RewriteRule .* - [E=HTTP_AUTHORIZATION:%{HTTP:Authorization}]
RewriteBase /
RewriteRule ^index\.php$ - [L]
RewriteCond %{REQUEST_FILENAME} !-f
RewriteCond %{REQUEST_FILENAME} !-d
RewriteRule . /index.php [L]

# END WordPress
`;
        }
        // fs.writeFileSync(`${zipTargetPath}/.htaccess`, htaccess, 'utf8');
        const rootDirectory = getRootDirectory();
        shellCd(zipTargetPath);
        shellExec(`git init && git add . && git commit -m "update"`);
        shellCd(rootDirectory);
        fs.rmSync(`./public/${tmpFolderExtract}`, { recursive: true, force: true });
      })();
      break;
    case 'nginx':
      // https://www.digitalocean.com/community/tutorials/how-to-install-linux-nginx-mariadb-php-lemp-stack-on-debian-10
      // https://github.com/pothi/wordpress-nginx/
      // https://github.com/digitalocean/nginxconfig.io
      break;
    default:
      throw new Error(`Program not found: ${program}`);
  }
  logger.info(`success: ${program} on ${os}`);
} catch (error) {
  logger.error(error, error.stack);
}
