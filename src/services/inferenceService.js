const tf = require("@tensorflow/tfjs-node");
const InputError = require("../exceptions/InputError");

async function predictClassification(model, image) {
  try {
    const tensor = tf.node
      .decodeJpeg(image)
      .resizeNearestNeighbor([224, 224])
      .expandDims()
      .toFloat();

    const prediction = model.predict(tensor);
    const score = await prediction.data();
    const confidenceScore = Math.max(...score) * 100;

    const classes = ["Cancer", "Non-cancer"];

    
    const classResult = tf.argMax(prediction, 1).dataSync()[0];
    let result = classes[classResult];

    let suggestion;

    if (confidenceScore > 50){
      result = 'Cancer';
      suggestion =
        "Segera periksa ke dokter!";
    } else {
      result = 'Non-cancer';
      suggestion =
        "Selamat Anda Sehat!";
    } 

    // let suggestion;

    // if (label === "Cancer") {
    //   suggestion =
    //     "Segera periksa ke dokter!";
    // }

    // if (label === "Non-cancer") {
    //   suggestion =
    //     "Selamat Anda Sehat!";
    // }
    return {result, suggestion};
  } catch (error) {
    throw new InputError(`Terjadi kesalahan input: ${error.message}`);
  }
}

module.exports = predictClassification;
