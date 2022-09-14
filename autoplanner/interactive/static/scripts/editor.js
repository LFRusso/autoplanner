function get_score(x, y) {
    let type = map[x + width * (height-(y+1))];
    let norm_accessibility = norm_accessibilities[x + width * (height-(y+1))];
    let mesh_distance = mesh_distances[x + width * (height-(y+1))];
    let K = view_radius[x + width * (height-(y+1))];

    let res = 0;
    let com = 0;
    let ind = 0;
    let rec = 0;
    for(let i=Math.max(x-K,0); i<Math.min(x+K+1, width); i++) {
        for(let j=Math.max(y-K,0); j<Math.min(y+K+1, height); j++) {
            switch(map[i + width * (height-(j+1))]) {
                case 1:
                    res += 1;
                    break;
                case 2:
                    com += 1;
                    break;
                case 3:
                    ind += 1;
                    break;
                case 4:
                    rec += 1;
                    break;
            }
        }
    }
    let score = 0
    switch(type) {
        case 1:
            res -= 1;
            score = norm_accessibility*(weights["Whh"]*res + weights["Whc"]*com + weights["Whi"]*ind + weights["Whr"]*rec)/K**2;
            break;
        case 2:
            com -= 1;
            let commercial_balance = Math.exp(-Math.pow(com - res,2)/K**2) - Math.exp(-Math.pow(res,2)/K**2);
            score = norm_accessibility*(weights["Wch"]*res/K**2 + weights["Wcc"]*commercial_balance);
            break;
        case 3:
            ind -= 1;
            score = Math.exp(-0.1*mesh_distance)*(weights["Wii"]*ind)/K**2
            break;
    }

    return score;
}

function update_cell(x, y, new_type) {
    map[x + width * (height-(y+1))] = parseInt(new_type)
    let K = view_radius[x + width * (height-(y+1))];

    for(let i=Math.max(x-K,0); i<Math.min(x+K+1, width); i++) {
        for(let j=Math.max(y-K,0); j<Math.min(y+K+1, height); j++) {
            scores[i + width * (height-(j+1))] = get_score(i, j);
        }
    }


    d3.select(`[id="map(${x},${y})"]`).remove()
    svg.append("rect")
    .attr("x", cell_size*x)
    .attr("y", cell_size*y)
    .attr("width", cell_size)
    .attr("height", cell_size)
    .attr("fill", (type=new_type) => {
        if(type==-1) return 'white';
        if(type==0) return 'black';
        if(type==1) return 'blue';
        if(type==2) return 'yellow';
        if(type==3) return 'red';
        if(type==4) return 'green';
        else return 'white';
    })
    .attr('stroke', 'gray')
    .on('mouseover', function (d, i) {
        d3.select(this).transition()
            .duration('50')
            .attr('opacity', '.5')
            .attr('stroke', 'red')
    })
    .on('mouseout', function (d, i) {
        d3.select(this).transition()
            .duration('50')
            .attr('opacity', '1')
            .attr('stroke', 'gray')
    })
    .on("click", (e) => {
        display_cell(x, y);
    });    
    
    display_cell(x, y);
    get_reward();
}

function get_reward() {
    let reward = scores.reduce((a, b) => a+b, 0);
    d3.select("#reward")
        .text("Reward: " + reward);

    return reward
}

function draw() {
    svg.selectAll('*').remove();

    svg.attr("width", cell_size*(width+1)/10)
    .attr("height", cell_size*(height+1)/10)

    for(let i=0; i<width; i++) {
        for(let j=0; j < height; j++) {
            svg.append("rect")
                .attr("x", cell_size*i)
                .attr("y", cell_size*j)
                .attr("width", cell_size)
                .attr("height", cell_size)
                .attr("id", `map(${i},${j})`)
                .attr("fill", (type=map[i + width * (height-(j+1))]) => {
                    if(type==-1) return 'white';
                    if(type==0) return 'black';
                    if(type==1) return 'blue';
                    if(type==2) return 'yellow';
                    if(type==3) return 'red';
                    if(type==4) return 'green';
                    else return 'white';
                })
                .attr('stroke', 'gray')
                .on('mouseover', function (d, i) {
                    d3.select(this).transition()
                        .duration('50')
                        .attr('opacity', '.5')
                        .attr('stroke', 'red')
                })
                .on('mouseout', function (d, i) {
                    d3.select(this).transition()
                        .duration('50')
                        .attr('opacity', '1')
                        .attr('stroke', 'gray')
                })
                .on("click", (e) => {
                    display_cell(i, j);
                });
                
        }
    }

    get_reward();
}

function display_cell(x, y) {
    selected_x = x;
    selected_y = y;

    let type = map[x + width * (height-(y+1))];
    let score = scores[x + width * (height-(y+1))];
    let accessibility = accessibilities[x + width * (height-(y+1))];
    let norm_accessibility = norm_accessibilities[x + width * (height-(y+1))];
    let mesh_distance = mesh_distances[x + width * (height-(y+1))];
    let radius = view_radius[x + width * (height-(y+1))];

    d3.selectAll("#cell-info").remove();
    d3.select("#editor")
        .append("div")
        .attr("id","cell-info")
    d3.select("#cell-info")
        .html(`
        <strong>Cell (${x},${y})</strong>
        <br> 
        Type: 
        ${type==0?`Road`:`
        <select id="type-opt">
            <option ${type==-1 ? `selected` : ''} value="-1">Undeveloped</option>
            <option ${type==1 ? `selected` : ''} value="1">Residential</option>
            <option ${type==2 ? `selected` : ''} value="2">Commercial</option>    
            <option ${type==3 ? `selected` : ''} value="3">Industial</option>    
            <option ${type==4 ? `selected` : ''} value="4">Recreational</option>    
        </select>`}
        <hr>
        <ul>
            <li>accessibility: `+ accessibility +`</li>
            <li>norm_accessibility: `+ norm_accessibility +`</li>
            <li>mesh_distance: `+ mesh_distance +`</li>
            <li>view_radius: `+ radius +`</li>
            <li>score: `+ score +`</li>
        </ul>
        `)

        let type_opt = d3.select("#type-opt")
        type_opt.on("change", function(e) {
            let new_type = d3.select(this).property('value');
            update_cell(x, y, new_type);
          })
}




let svg = d3.select("#fig");
let cell_size = 10;
let selected_x = null;
let selected_y = null;

// Actions
document.addEventListener("keydown", function(e) {
    if(e.key == '+'){
        cell_size += 1;
        draw();
    } 
    if(e.key == '-') {
        cell_size -= 1;
        draw();
        display_cell(selected_x, selected_y);
    }
  });

document.getElementById("reset-btn").addEventListener("click", function() {
    for(let i=0; i<width; i++) {
        for(let j=0; j < height; j++) {
            if(map[i + width * (height-(j+1))]!=original_map[i + width * (height-(j+1))]) {
                update_cell(i,j, original_map[i + width * (height-(j+1))]);
            }
        }
    }
});


d3.select("#weights")
    .attr("value",JSON.stringify(weights))

///

draw()