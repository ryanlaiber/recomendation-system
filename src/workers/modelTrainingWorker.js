import "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";
import { workerEvents } from "../events/constants.js";

console.log("Model training worker initialized");
let _globalCtx = {};
let _model = null;

const WEIGHTS = {
  category: 0.4,
  color: 0.3,
  price: 0.2,
  age: 0.1,
};

const normalize = (value, min, max) => (value - min) / (max - min || 1);

function makeContext(catalog, users) {
  const ages = users.map((u) => u.age);
  const prices = catalog.map((c) => c.price);

  const minAge = Math.min(...ages);
  const minPrice = Math.min(...prices);

  const maxAge = Math.max(...ages);
  const maxPrice = Math.max(...prices);

  const colors = [...new Set(catalog.map((c) => c.color))];
  const categories = [...new Set(catalog.map((c) => c.category))];

  const colorIndex = Object.fromEntries(
    colors.map((color, index) => {
      return [color, index];
    }),
  );

  const categoryIndex = Object.fromEntries(
    categories.map((category, index) => {
      return [category, index];
    }),
  );

  const midAge = (minAge + maxAge) / 2;
  const ageSums = {};
  const ageCounts = {};

  users.forEach((user) => {
    user.purchases.forEach((p) => {
      const productName = p.name;
      ageSums[productName] = (ageSums[productName] || 0) + user.age;
      ageCounts[productName] = (ageCounts[productName] || 0) + 1;
    });
  });

  const productAvgAgeNorm = Object.fromEntries(
    catalog.map((product) => {
      const avg = ageCounts[product.name]
        ? ageSums[product.name] / ageCounts[product.name]
        : midAge;

      return [product.name, normalize(avg, minAge, maxAge)];
    }),
  );

  return {
    catalog,
    users,
    colorIndex,
    categoryIndex,
    minAge,
    maxAge,
    minPrice,
    maxPrice,
    numCategories: categories.length,
    numColors: colors.length,
    dimensions: 2 + categories.length + colors.length,
    productAvgAgeNorm,
  };
}

function encodeUser(user, context) {
  if (user.purchases.length) {
    return tf
      .stack(user.purchases.map((product) => encodeProduct(product, context)))
      .mean(0)
      .reshape([1, context.dimensions]);
  }

  return tf
    .concat1d([
      tf.zeros([1]),
      tf.tensor1d([
        normalize(user.age, context.minAge, context.maxAge) * WEIGHTS.age,
      ]),
      tf.zeros([context.numCategories]),
      tf.zeros([context.numColors]),
    ])
    .reshape([1, context.dimensions]);
}

function createTrainingData(context) {
  const inputs = [];
  const labels = [];
  context.users
    .filter((u) => !!u.purchases?.length)
    .forEach((user) => {
      const userVector = encodeUser(user, context).dataSync();
      context.catalog.forEach((product) => {
        const productVector = encodeProduct(product, context).dataSync();

        const label = user.purchases.some((p) => p.name === product.name)
          ? 1
          : 0;
        inputs.push([...userVector, ...productVector]);
        labels.push(label);
      });
    });

  return {
    xs: tf.tensor2d(inputs),
    ys: tf.tensor2d(labels, [labels.length, 1]),
    inputDimention: context.dimensions * 2,
  };
}

const oneHotWeighted = (index, length, weight) =>
  tf.oneHot(index, length).cast("float32").mul(weight);

function encodeProduct(product, context) {
  const price = tf.tensor1d([
    normalize(product.price, context.minPrice, context.maxPrice) *
      WEIGHTS.price,
  ]);

  const age = tf.tensor1d([context.productAvgAgeNorm[product.name] || 0.5]);

  const category = oneHotWeighted(
    context.categoryIndex[product.category],
    context.numCategories,
    WEIGHTS.category,
  );

  const color = oneHotWeighted(
    context.colorIndex[product.color],
    context.numColors,
    WEIGHTS.color,
  );

  return tf.concat1d([price, age, category, color]);
}

async function configureNeuralNetAndTrain(trainData) {
  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      inputShape: [trainData.inputDimention],
      units: 128,
      activation: "relu",
    }),
  );

  model.add(
    tf.layers.dense({
      units: 64,
      activation: "relu",
    }),
  );

  model.add(
    tf.layers.dense({
      units: 32,
      activation: "relu",
    }),
  );

  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  await model.fit(trainData.xs, trainData.ys, {
    epochs: 100,
    batchSize: 32,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        postMessage({
          type: workerEvents.trainingLog,
          epoch,
          loss: logs.loss,
          accuracy: logs.acc,
        });
      },
    },
  });

  return model;
}

async function trainModel({ users }) {
  console.log("Training model with users:", users);
  postMessage({
    type: workerEvents.progressUpdate,
    progress: { progress: 50 },
  });
  const catalog = await (await fetch("/data/products.json")).json();

  const context = makeContext(catalog, users);

  context.productVectors = catalog.map((product) => {
    return {
      name: product.name,
      meta: { ...product },
      vector: encodeProduct(product, context).dataSync(),
    };
  });

  _globalCtx = context;

  const trainData = createTrainingData(context);

  _model = await configureNeuralNetAndTrain(trainData);
  postMessage({
    type: workerEvents.trainingLog,
    epoch: 1,
    loss: 1,
    accuracy: 1,
  });

  postMessage({
    type: workerEvents.progressUpdate,
    progress: { progress: 100 },
  });
  postMessage({ type: workerEvents.trainingComplete });
}

function recommend(user, ctx) {
  if (!_model) return;

  const context = _globalCtx;
  const model = _model;

  const userVector = encodeUser(user, _globalCtx).dataSync();

  const input = context.productVectors.map((v) => [...userVector, ...v.vector]);
  console.log("will recommend for user:", user);

  const inputTensor = tf.tensor2d(input);

  const predictions = model.predict(inputTensor);

  const scores = predictions.dataSync();

  const recomendations = context.productVectors.map((entry, index) => {
    return {
      ...entry.meta,
      name: entry.name,
      score: scores[index],
    };
  });

  const sortedItens = recomendations.sort((a, b) => b.score - a.score);

  postMessage({
    type: workerEvents.recommend,
    user,
    recommendations: sortedItens,
  });
}

const handlers = {
  [workerEvents.trainModel]: trainModel,
  [workerEvents.recommend]: (d) => recommend(d.user, _globalCtx),
};

self.onmessage = (e) => {
  const { action, ...data } = e.data;
  if (handlers[action]) handlers[action](data);
};
