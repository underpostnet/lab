import { endpointFactory } from '../../components/core/CommonJs.js';
import { loggerFactory } from '../../components/core/Logger.js';
import { getProxyPath } from '../../components/core/VanillaJs.js';

const logger = loggerFactory({ url: `${endpointFactory(import.meta)}-service` });

const proxyPath = getProxyPath();

const endpoint = endpointFactory(import.meta);

const API_BASE = `http://${location.host}${proxyPath}api${endpoint}`;

logger.info('Load service', API_BASE);

const FileService = {
  post: (body) =>
    new Promise((resolve, reject) =>
      fetch(API_BASE, {
        method: 'POST',
        // headers: {
        //   // 'Content-Type': 'application/json',
        //   // 'Authorization': ''
        // },
        body,
      })
        .then(async (res) => {
          return await res.json();
        })
        .then((res) => {
          logger.info(res);
          return resolve(res);
        })
        .catch((error) => {
          logger.error(error);
          return reject(error);
        })
    ),
};

export { FileService };