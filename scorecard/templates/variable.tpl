{% extends "html.tpl" %}
{% block before_style %}
<style type="text/css">
.ext {
    width: 50%;
    height: 10px;    
}
.ext.neg {
    border-right: 1px solid black;
    
}
.ext.pos {
    margin-left: 50%;
    border-left: 1px solid black;
}

.ext.neg .bar {
  float: right;
  background-color: lightcoral;
}

.ext.pos .bar {
  background-color: lightseagreen;  
}

.bar {
    height: 100%;
}
</style>
{% endblock before_style %}
{% block table %}
<table><tr><td><h3>{{ variable_name }}</h3></td><td>Step: {{ variable_step }}</td></tr></table>
{{ super() }}
{% endblock table %}
