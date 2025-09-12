// https://stackoverflow.com/questions/35769144/dynamically-adding-cases-to-a-switch

// you can create a new entry with this function
export function add_on(callbacks, _case, fn) {
  callbacks[_case] = callbacks[_case] || [];
  callbacks[_case].push(fn);
  return callbacks;
}

// this function work like switch (value)
//To make the name shorter you can name it `cond` (like in Scheme)
export function pseudoSwitch(callbacks, _case, args) {
  if (callbacks[_case]) {
    callbacks[_case].forEach(function (fn) {
      fn(args);
    });
  }
}
