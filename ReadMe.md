install minikube
install kubectl

minikube start --driver=docker

eval $(minikube docker-env) 

docker build -t ml_fastapi_docker .

kubectl apply -f k8s/deployment.yaml

kubectl apply -f k8s/service.yaml

kubectl get pods
