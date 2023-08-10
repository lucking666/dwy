// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import App from './App'
import router from './router'
import {makeRequest,uploadRequest} from './api/request'
import ElementUI from 'element-ui'
import 'element-ui/lib/theme-chalk/index.css'
import store from './vuex/store'

Vue.use(ElementUI)



Vue.config.productionTip = false;
Vue.prototype.$download=makeRequest;
Vue.prototype.$upload=uploadRequest;

/* eslint-disable no-new */
new Vue({
  el: '#app',
  store,
  router,
  components: { App },
  template: '<App/>'
})
