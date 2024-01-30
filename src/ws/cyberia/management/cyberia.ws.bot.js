import { CyberiaBiomeModel } from '../../../api/cyberia-biome/cyberia-biome.model.js';
import { CyberiaWorldModel } from '../../../api/cyberia-world/cyberia-world.model.js';
import { getId, random, range } from '../../../client/components/core/CommonJs.js';
import { BaseElement, getRandomAvailablePosition } from '../../../client/components/cyberia/CommonCyberia.js';
import pathfinding from 'pathfinding';

const CyberiaWsBotManagement = {
  element: {},
  pathfinding: {},
  worlds: [],
  instance: function (wsManagementId = '') {
    this.element[wsManagementId] = {};
    this.pathfinding.finder = new pathfinding.AStarFinder({
      allowDiagonal: true, // enable diagonal
      dontCrossCorners: true, // corner of a solid
      heuristic: pathfinding.Heuristic.chebyshev,
    });
    (async () => {
      this.worlds = await CyberiaWorldModel.find();
      this.biomes = await CyberiaBiomeModel.find();
      // const path = finder.findPath(
      //   element.render.x,
      //   element.render.y,
      //   x2,
      //   y2,
      //   new pathfinding.Grid(botMatrixCollision)
      // );

      for (const indexBot of range(0, 12)) {
        const bot = BaseElement().bot.main;
        // TODO: STANDARD WORLD TYPE
        // [1-6-3-5]
        // [1-2-3-4]
        const world = this.worlds.find((world) => world._id.toString() === bot.model.world._id);
        bot.model.world.face = (world.type === 'width' ? [1, 6, 3, 5] : [1, 2, 3, 4])[random(0, 3)];
        const biome = this.biomes.find(
          (biome) => biome._id.toString() === world.face[bot.model.world.face - 1].toString(),
        );
        const { x, y } = getRandomAvailablePosition({ biomeData: biome, element: bot });
        bot.x = x;
        bot.y = y;
        const id = getId(this.element[wsManagementId], 'bot-');
        this.element[wsManagementId][id] = bot;
      }
    })();
  },
};

export { CyberiaWsBotManagement };
