## Put and Delete--HTTP Verbs
## Working with API's--Json

from flask import Flask,jsonify,request

app=Flask(__name__)

## Initial Data in my to do list
items=[
    {"Id":1,"name":"Item1","description":"THsi is item 1"},
    {"Id":2,"name":"Item2","description":"THsi is item 2"}
]

app.route('/')
def home():
    return "welcome to the sample TO DO List app"

## GET: REtreive all the items
@app.route('/items',methods=["GET"])
def get_items():
    return jsonify(items)
## get:Rrtrive a specfic items by id
@app.route('/items/<int:item_id>',methods=['GET'])
def getitem(Item_Id):
    item=next((item for item in items if item["id"]==Item_Id),None)
    if item is None:
        return jsonify({"Erro:Item not found"})
    return  jsonify(item)

## Post: create a new task-API
@app.route('/items',methods=['POST'])
def create_item():
    if not request.json or not 'name' in request.json:
        return jsonify({"Error:Item not found"})
    new_item={
        "id":items[-1]["id"]+1 if items else 1,
        "name":request.json['name'],
        "descrpition":request.json["description"]
    }
    items.append(new_item)
    return jsonify(new_item)

## Put: Update an exixsting item
@app.route('/items/<int:item_id>',methods=['PUT'])
def update_item(item_id):
    item=next((item for item in items if item["id"]==item_id),None)
    if item is None:
        return jsonify({"Erro:Item not found"})
    item['name']=request.json.get('name',item['name'])
    item['description']=request.json.get('description',item['description'])
    return jsonify(item)

## Delete:delete an item
@app.route('/items/<int:item_id>',methods=['DELETE'])
def delete_item(item_id):
    global items
    items=[item for item in items if item["id"]!=item_id]
    return jsonify({"result":"Item deletd"})
    
if __name__=="___main__":
    app.run(debug=True)