import { Account } from '../core/Account.js';
import { BtnIcon } from '../core/BtnIcon.js';
import { getId, newInstance } from '../core/CommonJs.js';
import { Css, Themes } from '../core/Css.js';
import { EventsUI } from '../core/EventsUI.js';
import { LogIn } from '../core/LogIn.js';
import { LogOut } from '../core/LogOut.js';
import { Modal } from '../core/Modal.js';
import { SignUp } from '../core/SignUp.js';
import { Translate } from '../core/Translate.js';
import { s } from '../core/VanillaJs.js';
import { Elements } from './Elements.js';
import Sortable from 'sortablejs';
import { RouterNexodev } from './RoutesNexodev.js';

const Menu = {
  Data: {},
  Render: async function () {
    const id = getId(this.Data, 'menu-');
    this.Data[id] = {};
    const RouterInstance = RouterNexodev();
    const { NameApp } = RouterInstance;
    const { barConfig } = await Themes[Css.currentTheme]();
    await Modal.Render({
      id: 'modal-menu',
      html: html`
        <div class="fl menu-btn-container">
          ${await BtnIcon.Render({
            class: 'in fll main-btn-menu main-btn-home',
            label: 'Home',
            // style: 'display: none',
            attrs: `data-id="0"`,
          })}
          ${await BtnIcon.Render({
            class: 'in fll main-btn-menu main-btn-log-in',
            label: this.renderMenuLabel({ img: 'log-in.png', text: Translate.Render('log-in') }),
            attrs: `data-id="1"`,
          })}
          ${await BtnIcon.Render({
            class: 'in fll main-btn-menu main-btn-sign-up',
            label: this.renderMenuLabel({ img: 'sign-up.png', text: Translate.Render('sign-up') }),
            attrs: `data-id="2"`,
          })}
          ${await BtnIcon.Render({
            class: 'in fll main-btn-menu main-btn-log-out',
            label: this.renderMenuLabel({ img: 'log-out.png', text: Translate.Render('log-out') }),
            attrs: `data-id="3"`,
            style: 'display: none',
          })}
          ${await BtnIcon.Render({
            class: 'in fll main-btn-menu main-btn-account',
            label: this.renderMenuLabel({ img: 'account.png', text: Translate.Render('account') }),
            style: 'display: none',
            attrs: `data-id="4"`,
          })}
        </div>
      `,
      barConfig: newInstance(barConfig),
      title: NameApp,
      // titleClass: 'hide',
      mode: 'slide-menu',
    });

    this.Data[id].sortable = Modal.mobileModal()
      ? null
      : new Sortable(s(`.menu-btn-container`), {
          animation: 150,
          group: `menu-sortable`,
          forceFallback: true,
          fallbackOnBody: true,
          store: {
            /**
             * Get the order of elements. Called once during initialization.
             * @param   {Sortable}  sortable
             * @returns {Array}
             */
            get: function (sortable) {
              const order = localStorage.getItem(sortable.options.group.name);
              return order ? order.split('|') : [];
            },

            /**
             * Save the order of elements. Called onEnd (when the item is dropped).
             * @param {Sortable}  sortable
             */
            set: function (sortable) {
              const order = sortable.toArray();
              localStorage.setItem(sortable.options.group.name, order.join('|'));
            },
          },
          // chosenClass: 'css-class',
          // ghostClass: 'css-class',
          // Element dragging ended
          onEnd: function (/**Event*/ evt) {
            // console.log('Sortable onEnd', evt);
            // console.log('evt.oldIndex', evt.oldIndex);
            // console.log('evt.newIndex', evt.newIndex);
            const slotId = Array.from(evt.item.classList).pop();
            // console.log('slotId', slotId);
            if (evt.oldIndex === evt.newIndex) s(`.${slotId}`).click();

            // var itemEl = evt.item; // dragged HTMLElement
            // evt.to; // target list
            // evt.from; // previous list
            // evt.oldIndex; // element's old index within old parent
            // evt.newIndex; // element's new index within new parent
            // evt.oldDraggableIndex; // element's old index within old parent, only counting draggable elements
            // evt.newDraggableIndex; // element's new index within new parent, only counting draggable elements
            // evt.clone; // the clone element
            // evt.pullMode; // when item is in another sortable: `"clone"` if cloning, `true` if moving
          },
        });

    EventsUI.onClick(`.main-btn-sign-up`, async () => {
      const { barConfig } = await Themes[Css.currentTheme]();
      await Modal.Render({
        id: 'modal-sign-up',
        route: 'sign-up',
        barConfig,
        title: this.renderViewTile({ img: 'sign-up.png', text: Translate.Render('sign-up') }),
        html: async () => await SignUp.Render({ idModal: 'modal-sign-up' }),
        handleType: 'bar',
        maximize: true,
        mode: 'view',
        slideMenu: 'modal-menu',
        RouterInstance,
      });
    });

    EventsUI.onClick(`.main-btn-log-out`, async () => {
      const { barConfig } = await Themes[Css.currentTheme]();
      await Modal.Render({
        id: 'modal-log-out',
        route: 'log-out',
        barConfig,
        title: this.renderViewTile({ img: 'log-out.png', text: Translate.Render('log-out') }),
        html: async () => await LogOut.Render(),
        handleType: 'bar',
        maximize: true,
        mode: 'view',
        slideMenu: 'modal-menu',
        RouterInstance,
      });
    });

    EventsUI.onClick(`.main-btn-log-in`, async () => {
      const { barConfig } = await Themes[Css.currentTheme]();
      await Modal.Render({
        id: 'modal-log-in',
        route: 'log-in',
        barConfig,
        title: this.renderViewTile({ img: 'log-in.png', text: Translate.Render('log-in') }),
        html: async () => await LogIn.Render(),
        handleType: 'bar',
        maximize: true,
        mode: 'view',
        slideMenu: 'modal-menu',
        RouterInstance,
      });
    });

    EventsUI.onClick(`.main-btn-account`, async () => {
      const { barConfig } = await Themes[Css.currentTheme]();
      await Modal.Render({
        id: 'modal-account',
        route: 'account',
        barConfig,
        title: this.renderViewTile({ img: 'account.png', text: Translate.Render('account') }),
        html: async () => await Account.Render({ idModal: 'modal-account', user: Elements.Data.user.main.model.user }),
        handleType: 'bar',
        maximize: true,
        mode: 'view',
        slideMenu: 'modal-menu',
        RouterInstance,
      });
    });

    s(`.main-btn-home`).onclick = () => {
      for (const idModal of Object.keys(Modal.Data)) {
        if (Modal.Data[idModal].options.route) s(`.btn-close-${idModal}`).click();
      }
      s(`.btn-close-modal-menu`).click();
    };
  },
  renderMenuLabel: ({ img, text }) => text,
  // html`<img
  //     class="abs center img-btn-square-menu"
  //     src="${getProxyPath()}assets/ui-icons/${img}"
  //   />
  //   <div class="abs center main-btn-menu-text">${text}</div>`,

  renderViewTile: ({ img, text }) => text,
  // html`<img
  //     class="abs img-btn-square-view-title"
  //     src="${getProxyPath()}assets/ui-icons/${img}"
  //   />
  //   <div class="in text-btn-square-view-title">${text}</div>`,
};

export { Menu };