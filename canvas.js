const div_txt = document.getElementById('div_txt');
const div_degree = document.getElementById('div_degree');
const canvas_holder = document.getElementById('canvas_holder');
const optimizer = tf.train.adam(0.1);
const poly_xs = tf.linspace(-1,1,100);
const poly_xs_array = poly_xs.dataSync();

var canvas_holder_width = canvas_holder.clientWidth;
var canvas_holder_height = canvas_holder.clientHeight;
var xs = tf.tensor1d([]);
var ys = tf.tensor1d([]);
var coefficients = [];
var current_degree = -1;

function polynomial(x) {
  let y = tf.zerosLike(x);
  for (var i = 0; i < coefficients.length; i++) {
    y = y.add(x.pow(tf.tensor(i).toInt()).mul(coefficients[i]));
  }
  return y;
}

const loss = (pred, label) => pred.sub(label).square().mean();

function setup() {
  frameRate(60);
  var canvas = createCanvas(canvas_holder_width, canvas_holder_height);
  canvas.parent('canvas_holder');
}

function mousePressed() {
  if (mouseY >= 0) {
    xs = tf.tidy(() => {
      return xs.concat(
        tf.scalar(map(mouseX, 0, canvas_holder_width, -1, 1)).reshape([1])
      );
    })
    ys = tf.tidy(() => {
      return ys.concat(
        tf.scalar(map(mouseY, 0, canvas_holder_height, -1, 1)).reshape([1])
      );
    })
    coefficients.push(tf.variable(tf.scalar(0.0)));
    current_degree += 1;
  }
}

function windowResized() {
  canvas_holder_width = canvas_holder.clientWidth;
  canvas_holder_height = canvas_holder.clientHeight;
  resizeCanvas(canvas_holder_width, canvas_holder_height);
}

function draw() {
  background(255);

  if (current_degree >= 0) {
    let tmp_xs = xs.dataSync();
    let tmp_ys = ys.dataSync();
    for (var i = 0; i < tmp_xs.length; i++) {
      let x = map(tmp_xs[i], -1, 1, 0, canvas_holder_width);
      let y = map(tmp_ys[i], -1, 1, 0, canvas_holder_height);
      if (canvas_holder_width >= canvas_holder_height) {
        r = canvas_holder_height/50.0;
      } else {
        r = canvas_holder_width/50.0;
      }
      circle(x,y,r);
    }

    poly_ys = tf.tidy(() => polynomial(poly_xs));
    poly_ys_array = poly_ys.dataSync();
    beginShape();
    noFill();
    for (var i = 0; i < poly_xs_array.length; i++) {
      let x = map(poly_xs_array[i], -1, 1, 0, canvas_holder_width);
      let y = map(poly_ys_array[i], -1, 1, 0, canvas_holder_height);
      vertex(x,y);
    }
    endShape();
    poly_ys.dispose();

    let current_loss = tf.tidy(
      () => {return loss(polynomial(xs), ys).dataSync()[0]});
    div_txt.innerText = `Loss: ${current_loss.toExponential(1)}`;
    div_degree.innerText = `Degree: ${current_degree}`;
    optimizer.minimize(() => loss(polynomial(xs), ys));
  } else {
    if (canvas_holder_width >= canvas_holder_height) {
      textSize(canvas_holder_height/20.0);
    } else {
      textSize(canvas_holder_width/20.0);
    }
    textAlign(CENTER,CENTER);
    text('click on white area to add points', canvas_holder_width/2.0,
      canvas_holder_height/2.0
    );
  }
}
