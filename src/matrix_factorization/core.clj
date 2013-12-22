(ns matrix-factorization.core
  (:require [incanter.core :refer [$ sum sq abs mmult sqrt trans to-vect]]))

(defn norm [m]
  (incanter.core/sqrt (incanter.core/sum (map incanter.core/sum (incanter.core/sq m)))))

(defn dot [v1 v2]
  (reduce + (map * v1 v2)))

(defn get-rating-error [r p q]
  (- r (dot p q)))

(defn make-matrix-index-combination [R]
  (for [i  (range  (count R))
        j  (range  (count (first R)))] [i j]))

(defn get-error
  ([R P Q beta] (+ (* (/ beta 2.0) (+ (norm P) (norm Q))) (get-error R P Q (make-matrix-index-combination R) 0.0)))
  ([R P Q combination error]
    (if combination
      (let [[i j] (first combination)
            r ($ i j R)]
        (if (zero? r)
          (recur R P Q (next combination) error)
          (recur R P Q (next combination) (+ error (incanter.core/sq (get-rating-error r ($ i P) ($ j Q)))))))
      error)))

(defn update-matrix-in [m ks f & args]
  (incanter.core/matrix (update-in (incanter.core/to-vect m) ks #(apply f % args))))

(defn update-matrices [P Q K i j alpha err]
  (letfn [(iter [P Q k]
            (if (< k K)
              (recur (update-matrix-in P [k i] + (* alpha 2 err ($ k j Q)))
                     (update-matrix-in Q [k j] + (* alpha 2 err ($ k i P)))
                     (inc k))
              [P Q]))]
    (iter P Q 0)))

(defn do-step
  ([R P Q K alpha beta]
    (do-step R P Q K alpha beta (make-matrix-index-combination R)))
  ([R P Q K alpha beta combination]
    (if combination
      (let [[i j] (first combination)
            r ($ i j R)]
         (if (zero? r)
           (recur R P Q K alpha beta (next combination))
           (let [err (get-rating-error r ($ i P) ($ j Q))
                 [new-P new-Q] (update-matrices P Q K i j alpha err)]
                 (recur R new-P new-Q K alpha beta (next combination)))))
     [P Q])))


(defn matrix-factorization-step
  ([R P Q K steps alpha beta threshold]
    (if steps
      (let [[new-P new-Q] (do-step R P Q K alpha beta)
            error (get-error R new-P new-Q beta) ]
        (if (< error threshold)
          [new-P new-Q]
          (recur R new-P new-Q K (next steps) alpha beta threshold)))
    [P Q])))

(defn random-matrix [x y]
  (incanter.core/matrix (take (* x y) (repeatedly rand)) y))

(defn matrix-factorization [R K step alpha beta threshold]
  (matrix-factorization-step R
                             (random-matrix K (count R))
                             (random-matrix K (count (first R)))
                             K
                             (range step)
                             alpha
                             beta
                             threshold))

(defn -main []
  (let [R (incanter.core/matrix [
                                 [5.0, 3.0, 0.0, 1.0],
                                 [4.0, 0.0, 0.0, 1.0],
                                 [1.0, 1.0, 0.0, 5.0],
                                 [1.0, 0.0, 0.0, 4.0],
                                 [0.0, 1.0, 5.0, 4.0],
                                 ])
        [P Q] (matrix-factorization R 2 5000 0.0002 0.02 0.001)]
    (prn (incanter.core/mmult (incanter.core/trans P) Q))))
