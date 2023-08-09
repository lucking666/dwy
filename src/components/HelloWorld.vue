<template>
  <div class="hello">
    <h1>{{ msg }}</h1>
    <button >按钮获取其他</button>
    <el-button @click="handleClick">el-button点击我</el-button>
    <el-upload
      class="upload-demo"
      drag
      action="https://jsonplaceholder.typicode.com/posts/"
      :accept="`application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`"
      multiple>
      <i class="el-icon-upload"></i>
      <div class="el-upload__text">将文件拖到此处，或<em>点击上传</em></div>
      <div class="el-upload__tip" slot="tip">只能上传xlsx文件</div>
    </el-upload>
    <div @click="handlearr">{{ arr }}</div>
    
  </div>
</template>

<script>
import async from 'async';

export default {
  name: 'HelloWorld',
  data () {
    return {
      msg: '解决冲突',
      a:false,
      arr:[1,2,3,4,5,6,7,8],
    }
  },
  created() {
    // 在组件的生命周期钩子函数中使用 async 方法
  },
  methods:{
    //导出值的时候使用
    async post_request(){
      const loading=this.$loading({
        lock:true,
        test:'加载中...',
        background:"rgba(0,0,0,0.7)",
      });
      let url='https:/hddb/djbfk';
      let data='sheneh';
      let firstparam=1;
      let secondparam=2;
      let query=`firstparam=${firstparam}&secondparam=${secondparam}`;
      const result=await this.$download(
        url,'post',data
      ).then(res=>{
        console.log('成功');
      }).catch(err=>{
        console.log('失败');
      })
    },
    //导入文件并上传
    uploadfile(){
      const loading=this.$loading({
        lock:true,
        test:'加载中...',
        background:"rgba(0,0,0,0.7)",
      });
      this.loading=true;
      let {file}=this.formModel;
      let formData=new FormData();
      formData.append("file",file);
      this.$upload(this.uploadurl,formData)
      .then((res)=>{
        console.log('导入成功');
      })
      .catch((err)=>{
        console.log('导入失败');
      })
    },
    handleClick(){
      this.a=!this.a;
      this.msg=this.a?"变成elementui的按钮值":'Welcome to Your Vue.js App';
    },
    handlearr(){
      console.log(typeof(this.arr));
      this.arr=this.arr.slice(0,3);
      let arr = [1,2,3]
      console.log(arr[Symbol.iterator]) 
    }
  },
}
</script>
<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
h1, h2 {
  font-weight: normal;
}
ul {
  list-style-type: none;
  padding: 0;
}
li {
  display: inline-block;
  margin: 0 10px;
}
a {
  color: #42b983;
}
</style>
