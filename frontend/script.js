const API_BASE="http://127.0.0.1:5000"

let mriFile=null
let ctFile=null

document.getElementById("mriInput").addEventListener("change",(e)=>{

mriFile=e.target.files[0]

})

document.getElementById("ctInput").addEventListener("change",(e)=>{

ctFile=e.target.files[0]

})


document.getElementById("predictBtn").addEventListener("click",async()=>{

const formData=new FormData()

formData.append("mri",mriFile)
formData.append("ct",ctFile)
formData.append("patient_id","Demo123")

const response=await fetch(`${API_BASE}/predict`,{
method:"POST",
body:formData
})

const blob=await response.blob()

const url=window.URL.createObjectURL(blob)

const a=document.createElement("a")

a.href=url
a.download="Stroke_Report.pdf"

a.click()

})