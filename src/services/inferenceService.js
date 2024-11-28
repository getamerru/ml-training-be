const tf = require('@tensorflow/tfjs-node');
const InputError = require('../exceptions/InputError');

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

        // Tentukan label berdasarkan confidenceScore
        const label = confidenceScore <= 50 ? 'Non-cancer' : 'Cancer';
        let suggestion;

        if (label === 'Cancer') {
            suggestion = "Segera periksa ke dokter!";
        } else if (label === 'Non-cancer') {
            suggestion = "Anda sehat!";
        }

        // Membuat dua entri data sesuai format yang diinginkan
        const data = [
            {
                id: "13e907b3-4213-42ad-b12b-b9b7e12eb90e", // ID untuk Cancer
                history: {
                    result: "Cancer", // Hasil Cancer
                    createdAt: new Date().toISOString(), // Tanggal saat ini
                    suggestion: "Segera periksa ke dokter!", // Pesan untuk Cancer
                    id: "13e907b3-4213-42ad-b12b-b9b7e12eb90e" // ID unik untuk history Cancer
                }
            },
            {
                id: "19555e44-9cc7-4bc4-98b9-732d69cac082", // ID untuk Non-cancer
                history: {
                    result: "Non-cancer", // Hasil Non-cancer
                    createdAt: new Date().toISOString(), // Tanggal saat ini
                    suggestion: "Anda sehat!", // Pesan untuk Non-cancer
                    id: "19555e44-9cc7-4bc4-98b9-732d69cac082" // ID unik untuk history Non-cancer
                }
            }
        ];

        // Menyusun response sesuai dengan format yang diinginkan
        return {
            status: "success",
            data: data
        };
    } catch (error) {
        throw new InputError('Terjadi kesalahan dalam melakukan prediksi');
    }
}

module.exports = predictClassification;
