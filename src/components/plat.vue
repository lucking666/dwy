<template>
  <div>
    <p>这是主界面</p>
    <el-button type="primary" @click="gofenlan ">上传<i class="el-icon-upload el-icon--right"></i></el-button>  
    <el-button type="primary" @click="gohello ">查看<i class="el-icon-upload el-icon--right"></i></el-button> 
    <router-view :key="$route.Path"></router-view>   
  
  </div>
</template>

<script>
import { RouterView } from 'vue-router';
import fenlan from './fenlan.vue'
import HelloWorld from './HelloWorld.vue';
export default {
    name: 'plat',
    data(){
      return{
        isfenlan:false,
        ishello:false,
      };
    },
    components: { fenlan, HelloWorld},
    methods:{
      gofenlan(){
        console.log('hhhhhh')
        console.log(this.$route.path)
        if(this.isfenlan){
          return ;
        }
        this.$router.push({ name: 'fenlan' });
        this.isfenlan=true;
        this.ishello=false;
      },
      gohello(){
        console.log('hhhhhh')
        console.log(this.$route.path)
        if(this.ishello){
          return ;
        }
        this.$router.push({ name: 'HelloWorld' });
        this.ishello=true;
        this.isfenlan=false;
      },
    },
}
</script>
<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
div{
  height: 100%;
  margin-top: 0px;
}
</style>
