
rf = Roboflow(api_key="gjyu744KYWAeykWBn91r")
project = rf.workspace("cbumdev").project("liver-tumor-gqtmi-xdyce")
version = project.version(1)
dataset = version.download("yolov8")
                
                
                

rf = Roboflow(api_key="gjyu744KYWAeykWBn91r")
project = rf.workspace("cbumdev").project("kidney-tumor-detection-wcxga-wdxs9")
version = project.version(1)
dataset = version.download("yolov8")
                