(ns matrix-factorization.core
  (:require [incanter.core :refer [$ sum sq abs mmult sqrt trans to-vect]]))

(defn norm [m]
  (incanter.core/sqrt (incanter.core/sum (mapv incanter.core/sum (incanter.core/sq m)))))

(defn dot [v1 v2]
  (reduce + (mapv * v1 v2)))

(defn get-rating-error [r p q]
  (- r (dot p q)))

(defn make-matrix-index-combination [R]
  (for [i  (range  (count R))
        j  (range  (count (first R)))] [i j]))

(def make-matrix-index-combination-memo
  (memoize make-matrix-index-combination))

(defn get-column [m i]
  (mapv #(nth % i) m))

(defn get-error [R P Q beta]
  (letfn [(iter [combination error]
    (if combination
      (let [[i j] (first combination)
            r (get-in R [i j])]
        (if (zero? r)
          (recur (next combination) error)
          (recur (next combination) (+ error (incanter.core/sq (get-rating-error r (get-column P i) (get-column Q j)))))))
      error))]
    (+ (iter (make-matrix-index-combination-memo R) 0.0)
      (* (/ beta 2.0)
         (+ (norm P) (norm Q))
;         (+ (incanter.core/sq (norm P)) (incanter.core/sq (norm Q)))
         ))))

;(defn update-matrix-in [m ks f & args]
;  (incanter.core/matrix (update-in (incanter.core/to-list m) ks #(apply f % args))))

(defn update-matrices [P Q K i j alpha err]
  (letfn [(iter [P Q k]
            (if (< k K)
              (recur (update-in P [k i] + (* alpha 2 err (get-in Q [k j])))
                     (update-in Q [k j] + (* alpha 2 err (get-in P [k i])))
                     (inc k))
              [P Q]))]
    (iter P Q 0)))


(defn do-step [R P Q K alpha beta]
  (letfn [(iter [P Q combination]
            (if combination
              (let [[i j] (first combination)
                    r (get-in R [i j]) ]
                (if (zero? r)
                  (recur P Q (next combination))
                  (let [[new-P new-Q] (update-matrices P Q K i j alpha (get-rating-error r (get-column P i) (get-column Q j)))]
                    (recur new-P new-Q (next combination)))))
              [P Q]))]
    (iter P Q (make-matrix-index-combination-memo R))))


(defn matrix-factorization-step [R P Q K step alpha beta threshold]
  (letfn [(iter [P Q s]
            (if (< s step)
              (let [[new-P new-Q] (do-step R P Q K alpha beta)]
                (if (< (get-error R new-P new-Q beta) threshold)
                  [new-P new-Q]
                  (recur new-P new-Q (inc s))))
              [P Q]))]
    (iter P Q 0))
  )

(defn random-matrix [x y]
  (mapv vec (partition y (take (* x y) (repeatedly rand)))))

(defn matrix-factorization [R K step alpha beta threshold]
  (matrix-factorization-step R
                             (random-matrix K (count R))
                             (random-matrix K (count (first R)))
                             K
                             step
                             alpha
                             beta
                             threshold))

(defn -main []
  (let [R [
           [5.0, 3.0, 0.0, 1.0],
           [4.0, 0.0, 0.0, 1.0],
           [1.0, 1.0, 0.0, 5.0],
           [1.0, 0.0, 0.0, 4.0],
           [0.0, 1.0, 5.0, 4.0],
           ]
        [P Q] (matrix-factorization R 2 5000 0.0002 0.02 0.001)]
    (prn (incanter.core/mmult (incanter.core/trans (incanter.core/matrix P)) (incanter.core/matrix Q)))))
