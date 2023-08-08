import axios from 'axios';

// 创建一个 axios 请求函数
export const makeRequest = async (url, method, data) => {
  try {
    const response = await axios({
      method: method,
      url: url,
      data: data
    });
    // 返回响应数据
    return response.data;
  } catch (error) {
    // 处理错误
    throw new Error(error.message);
  }
};
const service = axios.create({
    baseURL: 'https://api.example.com',
    timeout: 5000,
    headers: {
      'Content-Type': 'application/json',
    },
  });
  
//使用axios发送请求并导出值
export const downRequest = async (url, method, data) => {
    return new Promise((response,reject)=>{
        service({method:'post',url:url,responseType:"blob"})
        .then((res)=>{
            console.log(response.data);
            if(response.size!=0){
                let className='hhhhh';
                let downloadA=document.getElementsByClassName(className);
                if(downloadA&&downloadA.length>0){
                    document.body.removeChild(downloadA[0]);
                }
                let url=window.URL.createObjectURL(new Blob([response]));
                let downloadElement=document.createElement("a");
                downloadElement.className=className;
                downloadElement.style.display="none";
                downloadElement.href=url;
                downloadElement.setAttribute("download",filename?`filename.xlsx`:`excel.xlsx`);
                document.body.appendChild(downloadElement);
                downloadA[0].cliick();
                reslove(true);
            }else{
                PromiseRejectionEvent(response.data);
            }
        });
    });
};
//导入文件并发送数据到远程的请求
export const uploadRequest = (url, params) => {
  return service({
    method:"post",
    url:url,
    data:params,
    headers:{
      "Content-Type":"mutipart/form-data",
    }
  })
};

