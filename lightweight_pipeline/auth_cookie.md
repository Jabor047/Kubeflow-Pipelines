# Getting Authentication Cookie for a Kubeflow deployment without ALB

## Service port-forwarding
The first step is to port-forward the istio-ingressgateway service (if you've not done it already).

> kubectl port-forward svc/istio-ingressgateway -n istio-system 7777:80


## Assign the host and port to an enviroment varible
> SERVICE=localhost
>
> PORT=7777

## Getting the authentication service session
```Bash
> 1. curl -v http://${SERVICE}:${PORT}
Look for line with:
<a href="/dex/auth?client_id=kubeflow-oidc-authservice&amp;redirect_uri=%2Flogin%2Foidc&amp;response_type=code&amp;scope=profile+email+groups+openid&amp;state=STATE_VALUE">Found</a>.

>Copy the value represented by the "STATE_VALUE" placeholder and Assign it to a variable.
STATE=STATE_VALUE

> 2. curl -v "http://${SERVICE}:${PORT}/dex/auth?client_id=kubeflow-oidc-authservice&redirect_uri=%2Flogin%2Foidc&response_type=code&scope=profile+email+groups+openid&amp;state=${STATE}"
Look for the line with:
<a href="/dex/auth/local?req=REQ_VALUE">Found</a>

> Copy the value represented by the "REQ_VALUE" placeholder and Assign it to a variable.
REQ=REQ_VALUE

> 3. curl -v "http://${SERVICE}:${PORT}dex/auth/local?req=${REQ}" -H 'Content-Type: application/x-www-form-urlencoded' --data 'login=admin%40kubeflow.org&password=12341234'

> Change the Login value and password value to match your credentials

> 4. curl -v "http://${SERVICE}:${PORT}/dex/approval?req=${REQ}"
Look for the line with:
<a href="/login/oidc?code=CODE_VALUE&amp;state=STATE_VALUE">See Other</a>.

> Copy the value represented by the "CODE_VALUE" placeholder and Assign it to a variable.
CODE=CODE_VALUE

> 5. curl -v "http://${SERVICE}:${PORT}/login/oidc?code=${CODE}&amp;state=${STATE}"
Look for the line with:
set-cookie authservice_session=SESSION

>> Copy the value represented by the "SESSION" placeholder and that is the authservice_session_cookie.
```