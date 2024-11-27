const tf = require('@tensorflow/tfjs-node');
const loadGraphModel = require('@tensorflow/tfjs-converter').loadGraphModel;

async function loadModel() {
    // Ganti dengan URL model yang benar dari Google Cloud Storage
    const modelUrl = 'https://storage.googleapis.com/bucket-modelll/model/model.json';
    if (!modelUrl) {
        throw new Error('Model URL tidak boleh null, mohon periksa lagi');
    }
    const model = await loadGraphModel(modelUrl);
    return model;
}

// Pastikan kamu mengekspor loadModel
module.exports = { loadModel };
