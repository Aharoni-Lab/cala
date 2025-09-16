// https://stackoverflow.com/questions/35769144/dynamically-adding-cases-to-a-switch

// you can create a new entry with this function
export function add_on(callbacks, _case, obj) {
  callbacks[_case] = obj;
  return callbacks;
}

// this function work like switch (_case)
//To make the name shorter you can name it `cond` (like in Scheme)
export function pseudoSwitch(callbacks, _case) {
  if (callbacks[_case]) {
    return callbacks[_case];
  }
}
