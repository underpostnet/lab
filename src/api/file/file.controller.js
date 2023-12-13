import { endpointFactory } from '../../client/components/core/CommonJs.js';
import { ProviderFactoryDB } from '../../db/ProviderFactoryDB.js';

import { loggerFactory } from '../../server/logger.js';
import { FileModel } from './file.model.js';

const endpoint = endpointFactory(import.meta);

const logger = loggerFactory({ url: `api-${endpoint}-controller` });

const DataBaseProvider = {};

const POST = async (req, res, options) => {
  try {
    const { host, path } = options;
    await ProviderFactoryDB(options, endpoint, DataBaseProvider);
    const db = DataBaseProvider[`${host}${path}`];
    if (db) logger.info('success get db provider', options.db);

    const results = [];
    if (Array.isArray(req.files.file))
      for (const file of req.files.file) results.push(await new FileModel(file).save());
    else if (req.files.file) results.push(await new FileModel(req.files.file).save());

    if (results.length === 0)
      return res.status(400).json({
        status: 'error',
        message: 'empty or invalid files',
      });

    return res.status(200).json({
      status: 'success',
      data: results,
    });
  } catch (error) {
    logger.error(error, error.stack);
    return res.status(500).json({
      status: 'error',
      message: error.message,
    });
  }
};

const GET = async (req, res, options) => {
  try {
    const { host, path } = options;
    await ProviderFactoryDB(options, endpoint, DataBaseProvider);
    const db = DataBaseProvider[`${host}${path}`];
    if (db) logger.info('success get db provider', options.db);

    // console.log('req.params', req.params);
    // console.log('req.query', req.query);
    // console.log('req.body', req.body);

    let result = {};
    switch (req.params.id) {
      case 'all':
        result = await FileModel.find();
        break;

      default:
        result = await FileModel.find({ _id: req.params.id });
        break;
    }

    // throw { message: 'error test' };
    return res.status(200).json({
      status: 'success',
      data: result,
    });
  } catch (error) {
    logger.error(error, error.stack);
    return res.status(500).json({
      status: 'error',
      message: error.message,
    });
  }
};

const DELETE = async (req, res, options) => {
  try {
    const { host, path } = options;
    await ProviderFactoryDB(options, endpoint, DataBaseProvider);
    const db = DataBaseProvider[`${host}${path}`];
    if (db) logger.info('success get db provider', options.db);

    let result = {};
    switch (req.params.id) {
      case 'all':
        break;

      default:
        result = await FileModel.findByIdAndDelete(req.params.id);
        break;
    }

    if (!result)
      return res.status(400).json({
        status: 'error',
        message: 'item not found',
      });

    return res.status(200).json({
      status: 'success',
      data: result,
      message: 'success-delete',
    });
  } catch (error) {
    logger.error(error, error.stack);
    return res.status(500).json({
      status: 'error',
      message: error.message,
    });
  }
};

export { POST, GET, DELETE };