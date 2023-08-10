import Vue from 'vue'
import Router from 'vue-router'
import HelloWorld from '@/components/HelloWorld'
import fenlan from '@/components/fenlan'
import plat from '@/components/plat'
Vue.use(Router)

export default new Router({
  mode:"history",
  base:"/",
  routes: [
    {
      path: '/',
      name: 'plat',
      component: plat,
      children:[
        {
          path: 'fenlan',
          name: 'fenlan',
          component: fenlan,
        },
        {
          path: 'HelloWorld',
          name: 'HelloWorld',
          component: HelloWorld,
        }
      ]
    },
  ]
})
